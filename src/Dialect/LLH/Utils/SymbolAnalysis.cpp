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

#include <cstddef>
#include <cstdint>
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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
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
    symbols_table_[symbol.str()] = symbol_int;
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

SymbolicIntOp SymbolAnalysis::buildNewSymbol() {
  auto module = symbol_module_->getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module));
  std::string symbol = "s" + std::to_string(next_symbol_id_);
  next_symbol_id_++;
  while (symbols_table_.count(symbol)) {
    symbol = "s" + std::to_string(next_symbol_id_);
    next_symbol_id_++;
  }
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto symbol_op =
      builder.create<SymbolicIntOp>(builder.getUnknownLoc(), symbol);
  _insertInModule(&builder, symbol_op);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildSymbol(const llvm::StringRef symbol) {
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto symbol_str = symbol.str();
  if (symbols_table_.count(symbol_str)) {
    return llvm::cast<SymbolicIntOp>(symbols_table_.at(symbol_str));
  }
  auto symbol_op =
      builder.create<SymbolicIntOp>(builder.getUnknownLoc(), symbol);
  _insertInModule(&builder, symbol_op);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildConstSymbol(size_t val) {
  std::string symbol = "c" + std::to_string(val);
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

Value SymbolAnalysis::addEncoding(Value value, size_t result_pos) {
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
      auto new_symbol = symbols_analysis->buildNewSymbol();
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
                                  llvm::ArrayRef<llvm::StringRef> symbols,
                                  size_t result_pos) {
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
      auto new_symbol = symbols_analysis->buildNewSymbol();
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

#define QIUCK_INSERT(table, key_exp, val_exp) \
  {                                           \
    std::string key = key_exp;                \
    std::string value = val_exp;              \
    auto& set = table[key];                   \
    if (!set.count(value)) set.insert(value); \
  }

#define Q_UNARY_INSERT(table, symbol, relation) \
  QIUCK_INSERT(table, symbol.str(), relation.str())

#define Q_BINARY_INSERT(table, symbol, lhs, rhs) \
  QIUCK_INSERT(table, symbol.str(), lhs.str() + "|" + rhs.str())

SymbolRelationOp SymbolAnalysis::buildSymbolRelation(
    const llvm::StringRef symbol, const llvm::StringRef relation,
    SymbolRelation relation_kind) {
  if (!hasSymbol(symbol)) {
    WRONG(llc::SymbolInfer) << "unknown symbol: " << symbol.str();
    return nullptr;
  }
  if (!hasSymbol(relation)) {
    WRONG(llc::SymbolInfer) << "unknown symbol: " << relation.str();
    return nullptr;
  }
  if (isConst(symbol) && isConst(relation)) return nullptr;
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto relation_op = builder.create<SymbolRelationOp>(
      builder.getUnknownLoc(), symbol, relation, relation_kind);
  _insertToSymbolModule(&builder, relation_op);
  switch (relation_kind) {
    case SymbolRelation::EQ:
      Q_UNARY_INSERT(EQ_table, symbol, relation)
      Q_UNARY_INSERT(EQ_table, relation, symbol)
      break;
    case SymbolRelation::GE:
      Q_UNARY_INSERT(GE_table, symbol, relation);
      Q_UNARY_INSERT(LE_table, relation, symbol);
      break;
    case SymbolRelation::GT:
      Q_UNARY_INSERT(GT_table, symbol, relation)
      Q_UNARY_INSERT(GE_table, symbol, relation)
      Q_UNARY_INSERT(LT_table, relation, symbol)
      Q_UNARY_INSERT(LE_table, relation, symbol)
      Q_UNARY_INSERT(NOTEQ_table, relation, symbol)
      Q_UNARY_INSERT(NOTEQ_table, symbol, relation)
      break;
    case SymbolRelation::LE:
      Q_UNARY_INSERT(LE_table, symbol, relation)
      Q_UNARY_INSERT(GE_table, relation, symbol)
      break;
    case SymbolRelation::LT:
      Q_UNARY_INSERT(LT_table, symbol, relation)
      Q_UNARY_INSERT(LE_table, symbol, relation)
      Q_UNARY_INSERT(GE_table, relation, symbol)
      Q_UNARY_INSERT(GT_table, relation, symbol)
      Q_UNARY_INSERT(NOTEQ_table, relation, symbol)
      Q_UNARY_INSERT(NOTEQ_table, symbol, relation)
    case SymbolRelation::NOTEQ:
      Q_UNARY_INSERT(NOTEQ_table, relation, symbol)
      Q_UNARY_INSERT(NOTEQ_table, symbol, relation)
      break;
  }
  return relation_op;
}

// TODO(lfr): 并查集重写
SymbolBinaryRelationOp SymbolAnalysis::buildSymbolRelation(
    const llvm::StringRef symbol, const llvm::StringRef relation_lhs,
    const llvm::StringRef relation_rhs, SymbolRelation relation_kind) {
  if (!hasSymbol(symbol)) {
    WRONG(llc::SymbolInfer) << "unknown symbol: " << symbol.str();
    return nullptr;
  }
  if (!hasSymbol(relation_lhs)) {
    WRONG(llc::SymbolInfer) << "unknown symbol: " << relation_lhs.str();
    return nullptr;
  }
  if (!hasSymbol(relation_rhs)) {
    WRONG(llc::SymbolInfer) << "unknown symbol: " << relation_rhs.str();
    return nullptr;
  }
  LLHPatternRewriter builder(symbol_module_->getContext());
  auto relation_op = builder.create<SymbolBinaryRelationOp>(
      builder.getUnknownLoc(), symbol, relation_lhs, relation_rhs,
      relation_kind);
  _insertToSymbolModule(&builder, relation_op);
  switch (relation_kind) {
    case SymbolRelation::Add:
      Q_BINARY_INSERT(Add_table, symbol, relation_lhs, relation_rhs);
      Q_BINARY_INSERT(Add_table, symbol, relation_rhs, relation_lhs)
      Q_BINARY_INSERT(Sub_table, relation_rhs, symbol, relation_lhs)
      Q_BINARY_INSERT(Sub_table, relation_rhs, symbol, relation_lhs)
      break;
    case SymbolRelation::Sub:
      Q_BINARY_INSERT(Sub_table, symbol, relation_lhs, relation_rhs);
      Q_BINARY_INSERT(Add_table, relation_lhs, relation_rhs, symbol)
      Q_BINARY_INSERT(Add_table, relation_lhs, symbol, relation_rhs)
    case SymbolRelation::Mul:
      Q_BINARY_INSERT(Mul_table, symbol, relation_lhs, relation_rhs);
      Q_BINARY_INSERT(Mul_table, symbol, relation_rhs, relation_lhs)
      Q_BINARY_INSERT(FloorDiv_table, relation_rhs, symbol, relation_lhs)
      Q_BINARY_INSERT(FloorDiv_table, relation_rhs, symbol, relation_lhs)
    case SymbolRelation::FloorDiv:
      Q_BINARY_INSERT(FloorDiv_table, symbol, relation_lhs, relation_rhs)
      break;
  }
  return relation_op;
}

#undef Q_BINARY_INSERT
#undef Q_UNARY_INSERT
#undef QIUCK_INSERT

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
  return symbols_table_.count(symbol.str());
}

#define PRINT_TABLE(table, relation)                              \
  for (auto& pair : table) {                                      \
    for (auto& relations : pair.second) {                         \
      INFO(llc::SymbolInfer)                                      \
          << pair.first << " [" << relation << "] " << relations; \
    }                                                             \
  }
void SymbolAnalysis::debugPrintSymbols() {
  for (auto symbols : symbols_table_) {
    INFO(llc::SymbolInfer) << "has symbol: " << symbols.first;
  }
  PRINT_TABLE(Add_table, "+")
  PRINT_TABLE(Sub_table, "-")
  PRINT_TABLE(Mul_table, "*")
  PRINT_TABLE(FloorDiv_table, "//")
  PRINT_TABLE(EQ_table, "==")
  PRINT_TABLE(GE_table, ">=")
  PRINT_TABLE(GT_table, ">")
  PRINT_TABLE(LE_table, "<=")
  PRINT_TABLE(LT_table, "<")
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
  auto symbol_module = getSymbolModule();
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
#undef PRINT_TABLE
}  // namespace mlir::llh
