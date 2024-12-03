//    Copyright 2024 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//

#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "symengine/basic.h"
#include "symengine/dict.h"
#include "symengine/integer.h"
#include "symengine/logic.h"
#include "symengine/printers.h"
#include "symengine/symbol.h"

namespace mlir::llh {
namespace {

llvm::StringRef BuildBinaryOpSymbolBind(Operation* op) {
  auto analysis = SymbolAnalysis::getInstance(op);
  auto lhs = op->getOperand(0);
  if (!SymbolAnalysis::hasSymbolAttr(lhs)) return SymbolAnalysis::UNKOW_SYMBOL;
  auto rhs = op->getOperand(1);
  if (!SymbolAnalysis::hasSymbolAttr(rhs)) return SymbolAnalysis::UNKOW_SYMBOL;
  llvm::SmallString<4> symbol;
  if (isConstIntegerValue(lhs) && isConstIntegerValue(rhs)) {
    size_t val = -1;
    if (isa<MulOp>(op)) {
      val = getConstIntegerValue(lhs) * getConstIntegerValue(rhs);
    } else if (isa<AddOp>(op)) {
      val = getConstIntegerValue(lhs) + getConstIntegerValue(rhs);
    } else if (isa<SubOp>(op)) {
      val = getConstIntegerValue(lhs) - getConstIntegerValue(rhs);
    } else if (isa<DivOp>(op)) {
      val = getConstIntegerValue(lhs) / getConstIntegerValue(rhs);
    } else {
      UNIMPLEMENTED(llc::SymbolInfer);
    }
    auto maybe_new_symbol = analysis->getOrBuildConstSymbol(val);
    llc::add_symbol_attr(op, maybe_new_symbol.getSymName());
    symbol = maybe_new_symbol.getSymName();
  } else {
    auto new_symbol = analysis->buildNewSymbol();
    llc::add_symbol_attr(op, new_symbol.getSymName());
    symbol = new_symbol.getSymName();
  }
  llh::SymbolRelation relation;
  if (isa<MulOp>(op)) {
    relation = SymbolRelation::Mul;
  } else if (isa<AddOp>(op)) {
    relation = SymbolRelation::Add;
  } else if (isa<SubOp>(op)) {
    relation = SymbolRelation::Sub;
  } else if (isa<DivOp>(op)) {
    relation = SymbolRelation::FloorDiv;
  } else {
    UNIMPLEMENTED(llc::SymbolInfer);
  }
  analysis->buildSymbolRelation(analysis->getSymbolAttr(op),
                                analysis->getSymbolAttr(lhs),
                                analysis->getSymbolAttr(rhs), relation);
  return symbol;
}

llvm::StringRef BuildShapedWithConstSymbolBind(Operation* op, Value shape_value,
                                               Value const_value) {
  if (!llc::hasEncoding(shape_value)) return SymbolAnalysis::UNKOW_SYMBOL;
  LLHPatternRewriter builder(op->getContext());
  auto shape_symbols = llc::getEncodingFrom(shape_value).getShapeSymbols();
  auto dim = llh::getConstIntegerValue(const_value);
  op->setAttr(llc::SymbolIntAttr, shape_symbols[dim]);
  return shape_symbols[dim].getValue();
}

llvm::StringRef BuildConstSymbolBind(Operation* op) {
  auto res = op->getResult(0);
  if (!llvm::isa<IndexType, IntegerType>(res.getType()))
    return SymbolAnalysis::UNKOW_SYMBOL;
  size_t val;
  if (isa<ConstantOp>(op)) {
    val = llvm::cast<IntegerAttr>(op->getAttr("value")).getInt();
  } else {
    UNIMPLEMENTED(llc::UTILITY);
  }
  auto symbol_analysis = SymbolAnalysis::getInstance(op);
  auto symbol_dim_op = symbol_analysis->getOrBuildConstSymbol(val);
  llc::add_symbol_attr(op, symbol_dim_op.getSymName());
  return symbol_dim_op.getSymName().str();
}

}  // namespace

std::mutex SymbolAnalysisManager::mutex_;
SymbolAnalysisManager& SymbolAnalysisManager::getInstance() {
  static SymbolAnalysisManager instance;
  return instance;
}

std::mutex SymbolAnalysis::mutex_;

StringRef SymbolAnalysis::UNKOW_SYMBOL = "UNKOW";

mlir::StringRef SymbolAnalysis::symbol_module_name = "__symbol__";

bool SymbolAnalysis::symbol_enable = true;

SymbolAnalysis::SymbolAnalysis(Operation* op) {
  ModuleOp module;
  if (llvm::isa<ModuleOp>(op)) {
    module = llvm::cast<ModuleOp>(op);
  } else if (llvm::isa<SymbolRelationOp, SymbolBinaryRelationOp>(op)) {
    auto symbol_module = op->getParentOfType<ModuleOp>();
    CHECK(llc::SymbolInfer, (symbol_module.getSymName()->str() ==
                             SymbolAnalysis::symbol_module_name.str()));
    module = symbol_module->getParentOfType<ModuleOp>();
  } else {
    module = op->getParentOfType<ModuleOp>();
  }
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module))
      << "must build an instance for a module";
  auto symbol_ints = module.getOps<SymbolicIntOp>();
  for (auto symbol_int : symbol_ints) {
    auto symbol = symbol_int.getSymName();
    symbol_op_table_[symbol.str()] = symbol_int;
  }
  auto symbol_module = SymbolTable::lookupNearestSymbolFrom(
      module.getOperation(),
      StringAttr::get(module->getContext(), symbol_module_name));
  if (!symbol_module) {
    LLHPatternRewriter builder(module);
    symbol_module =
        builder.create<ModuleOp>(module->getLoc(), symbol_module_name);
    module->getRegion(0).getBlocks().front().push_back(symbol_module);
  }
  CHECK(llc::SymbolInfer, symbol_module);
  symbol_module_ = symbol_module;
}
SymbolAnalysis::~SymbolAnalysis() {}

SymbolAnalysis* SymbolAnalysis::getInstance(Operation* op) {
  std::lock_guard<std::mutex> lock(SymbolAnalysisManager::mutex_);
  ModuleOp module;
  if (llvm::isa<ModuleOp>(op)) {
    module = llvm::cast<ModuleOp>(op);
    // for test
  } else if (llvm::isa<SymbolRelationOp, SymbolBinaryRelationOp>(op)) {
    auto symbol_module = op->getParentOfType<ModuleOp>();
    CHECK(llc::SymbolInfer, (symbol_module.getSymName()->str() ==
                             SymbolAnalysis::symbol_module_name.str()));
    module = symbol_module->getParentOfType<ModuleOp>();
  } else {
    module = op->getParentOfType<ModuleOp>();
  }
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module))
      << "must build an instance for a module";
  auto& analysis_map = SymbolAnalysisManager::getInstance().analysis_map_;
  if (!analysis_map.count(module) || analysis_map[module] == nullptr) {
    analysis_map[module] = new SymbolAnalysis(module);
  }
  return analysis_map[module];
}

SymbolAnalysis* SymbolAnalysis::getInstance(Value value) {
  std::lock_guard<std::mutex> lock(SymbolAnalysisManager::mutex_);
  auto module = value.getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module))
      << "must build an instance for a module";
  if (!SymbolAnalysisManager::getInstance().analysis_map_.contains(module)) {
    SymbolAnalysisManager::getInstance().analysis_map_[module] =
        new SymbolAnalysis(module);
  }
  return SymbolAnalysisManager::getInstance().analysis_map_[module];
}

bool SymbolAnalysis::cleanCache() {
  auto module = getRootModule();
  auto& manager = SymbolAnalysisManager::getInstance();
  manager.analysis_map_[module] = nullptr;
  return true;
}

bool SymbolAnalysis ::isExtraSymbolicInferOp(Operation* op) {
  return isa<DimOp>(op);
}

bool SymbolAnalysis ::isSymbolicInferOp(Operation* op) {
  return isExtraSymbolicInferOp(op) || isa<SymbolicInferShapeOpInterface>(op);
}

bool SymbolAnalysis::hasSymbolAttr(Operation* op) {
  if (!isSymbolicInferOp(op)) return false;
  return op->hasAttr(llc::SymbolIntAttr);
}
bool SymbolAnalysis::hasSymbolAttr(Value value) {
  auto op = value.getDefiningOp();
  return hasSymbolAttr(op);
}

llvm::StringRef SymbolAnalysis::getSymbolAttr(Operation* op) {
  if (!hasSymbolAttr(op)) return UNKOW_SYMBOL;
  return llvm::cast<FlatSymbolRefAttr>(op->getAttr(llc::SymbolIntAttr))
      .getValue();
}
llvm::StringRef SymbolAnalysis::getSymbolAttr(Value value) {
  if (isa<BlockArgument>(value)) return UNKOW_SYMBOL;
  auto op = value.getDefiningOp();
  return getSymbolAttr(op);
}
SymEngine::RCP<const SymEngine::Basic> SymbolAnalysis::getBasicSymbol(
    const llvm::StringRef symbol) {
  CHECK(llc::SymbolInfer, symbol_table_.contains(symbol.str()));
  return symbol_table_[symbol.str()];
}
SymbolicIntOp SymbolAnalysis::buildNewSymbol(
    const SymEngine::RCP<const SymEngine::Basic> symbol, AffineMap affine_map,
    llvm::ArrayRef<llvm::StringRef> relations, bool greater_zore) {
  auto module = symbol_module_->getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module));
  std::string symbol_name = "s" + std::to_string(next_symbol_id_);
  next_symbol_id_++;
  while (symbol_op_table_.count(symbol_name)) {
    symbol_name = "s" + std::to_string(next_symbol_id_);
    next_symbol_id_++;
  }
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto symbol_op =
      _insertNewSymbol(symbol_name, &builder, greater_zore, symbol);
  buildSymbolRelation(symbol_name, affine_map, relations);
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::buildNewSymbol(bool greater_zore) {
  auto module = symbol_module_->getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module));
  std::string symbol = "s" + std::to_string(next_symbol_id_);
  next_symbol_id_++;
  while (symbol_op_table_.count(symbol)) {
    symbol = "s" + std::to_string(next_symbol_id_);
    next_symbol_id_++;
  }
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto symbol_op = _insertNewSymbol(symbol, &builder, greater_zore);
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildSymbol(const llvm::StringRef symbol,
                                               bool greater_zore) {
  auto symbol_str = symbol.str();
  if (symbol_op_table_.count(symbol_str)) {
    return llvm::cast<SymbolicIntOp>(symbol_op_table_.at(symbol_str));
  }
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto symbol_op = _insertNewSymbol(symbol, &builder, greater_zore);
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildConstSymbol(size_t val) {
  llvm::SmallString<1> symbol("c" + std::to_string(val));
  return getOrBuildSymbol(symbol);
}

SymbolBindOp SymbolAnalysis::buildSymbolBindFromAttr(Value value,
                                                     OpBuilder* builder) {
  if (!hasSymbolAttr(value)) return nullptr;
  builder->setInsertionPointAfterValue(value);
  return builder->create<SymbolBindOp>(value.getLoc(), value,
                                       getSymbolAttr(value));
}

EncodingBindOp SymbolAnalysis::buildEncodingBindFrom(Value value,
                                                     OpBuilder* builder) {
  if (!llc::hasEncoding(value)) return nullptr;
  // builder->setInsertionPointAfterValue(value);
  auto encoding = llc::getEncodingFrom(value);
  auto encoding_bind = builder->create<EncodingBindOp>(
      value.getLoc(), ::mlir::TypeRange{}, value, encoding);
  encoding_bind->moveAfter(value.getDefiningOp());
  return encoding_bind;
}

void SymbolAnalysis::buildEncodingBindFrom(Operation* op, OpBuilder* builder) {
  for (auto res : op->getResults()) {
    buildEncodingBindFrom(res, builder);
  }
}

void SymbolAnalysis::unloadEncoding(Value value) {
  if (!llc::hasEncoding(value)) return;
  auto tensor = llc::getRankTensorFrom(value);
  auto new_tensor =
      RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  value.setType(new_tensor);
}

void SymbolAnalysis::unloadEncoding(Operation* op) {
  for (auto res : op->getResults()) {
    unloadEncoding(res);
  }
}

Value SymbolAnalysis::addEncoding(Value value) {
  IRRewriter builder(value.getContext());
  auto type = value.getType();
  if (!isa<RankedTensorType>(type)) return value;
  auto unencoding_tensor = llvm::cast<RankedTensorType>(type);
  auto try_get_encoding = unencoding_tensor.getEncoding();
  if (try_get_encoding) {
    auto has_encoding_attr = isa<EncodingAttr>(try_get_encoding);
    if (has_encoding_attr) return value;
  }
  auto symbols_analysis = SymbolAnalysis::getInstance(value);
  auto symbols = llvm::SmallVector<StringRef>();
  for (auto dim : unencoding_tensor.getShape()) {
    if (dim == ShapedType::kDynamic) {
      auto new_symbol = symbols_analysis->buildNewSymbol(true);
      symbols.push_back(new_symbol.getSymName());
    } else {
      auto const_symbol = symbols_analysis->getOrBuildConstSymbol(dim);
      symbols.push_back(const_symbol.getSymName());
    }
  }
  auto encoding = EncodingAttr::get(builder.getContext(), symbols);
  auto new_tensor_type =
      RankedTensorType::get(unencoding_tensor.getShape(),
                            unencoding_tensor.getElementType(), encoding);
  value.setType(new_tensor_type);
  return value;
}

Value SymbolAnalysis::addEncoding(Value value,
                                  llvm::ArrayRef<llvm::StringRef> symbols) {
  IRRewriter builder(value.getContext());
  auto type = value.getType();
  if (!isa<RankedTensorType>(type)) return value;
  auto unencoding_tensor = llvm::cast<RankedTensorType>(type);
  auto try_get_encoding = unencoding_tensor.getEncoding();
  if (try_get_encoding) {
    auto has_encoding_attr = isa<EncodingAttr>(try_get_encoding);
    if (has_encoding_attr) return value;
  }
  auto symbols_analysis = SymbolAnalysis::getInstance(value);
  auto rank = unencoding_tensor.getRank();
  auto shape = unencoding_tensor.getShape();
  auto new_symbols = SmallVector<StringRef>();
  for (int i = 0; i < rank; i++) {
    if (UNKOW_SYMBOL.str() != symbols[i].str()) {
      CHECK(llc::SymbolInfer, hasSymbol(symbols[i]));
      new_symbols.push_back(symbols[i]);
      continue;
    }
    auto dim = shape[i];
    if (dim == ShapedType::kDynamic) {
      auto new_symbol = symbols_analysis->buildNewSymbol(true);
      new_symbols.push_back(new_symbol.getSymName());
    } else {
      auto const_symbol = symbols_analysis->getOrBuildConstSymbol(dim);
      new_symbols.push_back(const_symbol.getSymName());
    }
  }
  auto encoding = EncodingAttr::get(builder.getContext(), new_symbols);
  auto new_tensor_type =
      RankedTensorType::get(unencoding_tensor.getShape(),
                            unencoding_tensor.getElementType(), encoding);
  value.setType(new_tensor_type);
  return value;
}

llvm::StringRef SymbolAnalysis::getOrBuildSymbolAttrFrom(Operation* op) {
  if (hasSymbolAttr(op)) return getSymbolAttr(op);
  if (isa<DimOp>(op)) {
    return BuildShapedWithConstSymbolBind(op, op->getOperand(0),
                                          op->getOperand(1));
  }
  if (isa<mlir::arith::ConstantIntOp, mlir::arith::ConstantOp,
          mlir::arith::ConstantIndexOp, llh::ConstantOp>(op)) {
    return BuildConstSymbolBind(op);
  }
  if (isa<DivOp, MulOp, SubOp, AddOp>(op)) {
    return BuildBinaryOpSymbolBind(op);
  }
  return SymbolAnalysis::UNKOW_SYMBOL;
}

llvm::StringRef SymbolAnalysis::getOrBuildSymbolAttrFrom(Value value) {
  auto op = value.getDefiningOp();
  return getOrBuildSymbolAttrFrom(op);
}

SymbolRelationOp SymbolAnalysis::buildSymbolRelation(
    const llvm::StringRef symbol, const llvm::StringRef relation,
    SymbolRelation relation_kind) {
  CHECK(llc::SymbolInfer, hasSymbol(symbol));
  CHECK(llc::SymbolInfer, hasSymbol(relation));
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto relation_op = builder.create<SymbolRelationOp>(
      builder.getUnknownLoc(), symbol, relation, relation_kind);
  _insertToSymbolModule(&builder, relation_op);
  return relation_op;
}

SymbolBinaryRelationOp SymbolAnalysis::buildSymbolRelation(
    const llvm::StringRef symbol, const llvm::StringRef relation_lhs,
    const llvm::StringRef relation_rhs, SymbolRelation relation_kind) {
  CHECK(llc::SymbolInfer, hasSymbol(symbol));
  CHECK(llc::SymbolInfer, hasSymbol(relation_lhs));
  CHECK(llc::SymbolInfer, hasSymbol(relation_rhs));
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto relation_op = builder.create<SymbolBinaryRelationOp>(
      builder.getUnknownLoc(), symbol, relation_lhs, relation_rhs,
      relation_kind);
  _insertToSymbolModule(&builder, relation_op);
  return relation_op;
}
SymbolRelationMapOp SymbolAnalysis::buildSymbolRelation(
    const llvm::StringRef symbol, AffineMap affine_map,
    llvm::ArrayRef<llvm::StringRef> relations) {
  CHECK(llc::SymbolInfer, hasSymbol(symbol));
  LLHPatternRewriter builder(symbol_module_->getContext());
  llvm::SmallVector<Attribute> attrs;
  for (auto relation : relations) {
    CHECK(llc::SymbolInfer, hasSymbol(relation));
    attrs.push_back(llvm::cast<Attribute>(
        FlatSymbolRefAttr::get(getOrBuildSymbol(relation))));
  }
  auto relation_attr = ArrayAttr::get(builder.getContext(), attrs);
  auto relation_op = builder.create<SymbolRelationMapOp>(
      builder.getUnknownLoc(), symbol, relation_attr, affine_map);
  _insertToSymbolModule(&builder, relation_op);
  return relation_op;
}

bool SymbolAnalysis::replaceSymbol(const llvm::StringRef old_symbol,
                                   const llvm::StringRef new_symbol) {
  CHECK(llc::SymbolInfer, hasSymbol(old_symbol));
  CHECK(llc::SymbolInfer, hasSymbol(new_symbol));
  if (old_symbol.str() == new_symbol.str()) return true;
  auto module = getRootModule();
  AttrTypeReplacer replacer;
  replacer.addReplacement([&old_symbol, &new_symbol](FlatSymbolRefAttr attr)
                              -> std::pair<Attribute, WalkResult> {
    if (attr.getValue().str() == old_symbol.str())
      return {FlatSymbolRefAttr::get(attr.getContext(), new_symbol),
              WalkResult::skip()};
    return {attr, WalkResult::skip()};
  });
  module->walk([&replacer](Operation* op) {
    replacer.replaceElementsIn(op, true, false, true);
  });
  return true;
}
ModuleOp SymbolAnalysis::getSymbolModule() const {
  return cast<ModuleOp>(symbol_module_);
}

ModuleOp SymbolAnalysis::getRootModule() const {
  return getSymbolModule().getOperation()->getParentOfType<ModuleOp>();
}

bool SymbolAnalysis::hasSymbol(const llvm::StringRef symbol) const {
  return symbol_op_table_.count(symbol.str());
}

void SymbolAnalysis::debugPrintSymbols() {
  for (auto& map : this->symbol_table_) {
    auto name = map.first;
    auto symbol = map.second;
    std::cout << name << ": " << SymEngine::ccode(*symbol.get()) << std::endl;
  }
}

void SymbolAnalysis::_insertInModule(LLHPatternRewriter* builder,
                                     Operation* op) const {
  ModuleOp module = getRootModule();
  CHECK(llc::SymbolInfer, module);
  auto& block = module->getRegion(0).getBlocks().front();
  op->remove();
  block.push_front(op);
}

void SymbolAnalysis::_insertToSymbolModule(LLHPatternRewriter* builder,
                                           Operation* op) const {
  ModuleOp module = llvm::cast_or_null<ModuleOp>(symbol_module_);
  CHECK(llc::MLIR, module);
  auto& block = module->getRegion(0).getBlocks().back();
  op->remove();
  block.push_front(op);
}

bool SymbolAnalysis::_isConst(Operation* op) {
  return isConst(getSymbolAttr(op));
}
bool SymbolAnalysis::_isConst(Value value) {
  return isConst(getSymbolAttr(value));
}
bool SymbolAnalysis::isConst(const llvm::StringRef name) {
  return name.starts_with("c");
}
int64_t SymbolAnalysis::_getConst(const llvm::StringRef name) {
  CHECK(llc::SymbolInfer, isConst(name));
  int64_t res;
  llvm::to_integer(name.substr(1), res);
  return res;
}

SymbolicIntOp SymbolAnalysis::_insertNewSymbol(
    const llvm::StringRef symbol_name, LLHPatternRewriter* builder,
    bool greater_zore) {
  auto symbol_str = symbol_name.str();
  auto symbol_op =
      builder->create<SymbolicIntOp>(builder->getUnknownLoc(), symbol_name);
  _insertInModule(builder, symbol_op);
  symbol_op_table_[symbol_str] = symbol_op;
  if (!isConst(symbol_name)) {
    auto symbol = SymEngine::symbol(symbol_str);
    symbol_table_[symbol_str] = symbol->rcp_from_this();
  } else {
    auto value = _getConst(symbol_name);
    auto symbol = SymEngine::integer(value);
    symbol_table_[symbol_str] = symbol->rcp_from_this();
  }
  if (!isConst(symbol_name) && greater_zore) {
    auto one = getOrBuildConstSymbol(1);
    buildSymbolRelation(symbol_name, one.getSymName(), SymbolRelation::GE);
  }
  return symbol_op;
};

SymbolicIntOp SymbolAnalysis::_insertNewSymbol(
    const llvm::StringRef symbol_name, LLHPatternRewriter* builder,
    bool greater_zore, const Symbol symbol) {
  auto symbol_str = symbol_name.str();
  auto symbol_op =
      builder->create<SymbolicIntOp>(builder->getUnknownLoc(), symbol_name);
  _insertInModule(builder, symbol_op);
  symbol_op_table_[symbol_str] = symbol_op;
  symbol_table_[symbol_str] = symbol->rcp_from_this();
  if (!isConst(symbol_name) && greater_zore) {
    auto one = getOrBuildConstSymbol(1);
    buildSymbolRelation(symbol_name, one.getSymName(), SymbolRelation::GE);
  }
  return symbol_op;
}
}  // namespace mlir::llh
