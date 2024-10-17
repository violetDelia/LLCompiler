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

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::llh {
namespace {

llvm::StringRef BuildBinaryOpSymbolBind(Operation* op) {
  auto analysis = SymbolAnalysis::getInstance(op);
  auto lhs = op->getOperand(0);
  if (!SymbolAnalysis::hasSymbolAttr(lhs)) return SymbolAnalysis::UNKOW_SYMBOL;
  auto rhs = op->getOperand(1);
  if (!SymbolAnalysis::hasSymbolAttr(rhs)) return SymbolAnalysis::UNKOW_SYMBOL;
  auto maybe_new_symbol = analysis->buildNewSymbolFrom(op->getResult(0));
  llc::add_symbol_attr(op, maybe_new_symbol.getSymName());
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
  analysis->buildSymbolRelation(op->getResult(0), lhs, rhs,
                                llh::SymbolRelation::Mul);
  analysis->buildSymbolRelation(op->getResult(0), rhs, lhs,
                                llh::SymbolRelation::Mul);
  return maybe_new_symbol.getSymName();
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
  auto symbol_dim_op = symbol_analysis->getOrBuildConstSymbolFrom(res, val);
  llc::add_symbol_attr(op, symbol_dim_op.getSymName());
  return symbol_dim_op.getSymName().str();
}

}  // namespace

SymbolAnalysisManager* SymbolAnalysisManager::instance_ =
    new (std::nothrow) SymbolAnalysisManager;

std::mutex SymbolAnalysisManager::mutex_;

std::mutex SymbolAnalysis::mutex_;

StringRef SymbolAnalysis::UNKOW_SYMBOL = "UNKOW";

mlir::StringRef SymbolAnalysis::symbol_module_name = "__symbol__";

bool SymbolAnalysis::symbol_enable = true;

SymbolAnalysis::SymbolAnalysis(Operation* op) {
  ModuleOp module;
  if (llvm::isa<ModuleOp>(op)) {
    module = llvm::cast<ModuleOp>(op);
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
  ModuleOp module;
  if (llvm::isa<ModuleOp>(op)) {
    module = llvm::cast<ModuleOp>(op);
  } else {
    module = op->getParentOfType<ModuleOp>();
  }
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module))
      << "must build an instance for a module";
  std::lock_guard<std::mutex> lock(mutex_);
  if (!SymbolAnalysisManager::instance_->analysis_map_.contains(module)) {
    SymbolAnalysisManager::instance_->analysis_map_[module] =
        new SymbolAnalysis(module);
  }
  return SymbolAnalysisManager::instance_->analysis_map_[module];
}

SymbolAnalysis* SymbolAnalysis::getInstance(Value value) {
  auto module = value.getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module))
      << "must build an instance for a module";
  std::lock_guard<std::mutex> lock(SymbolAnalysisManager::mutex_);
  if (!SymbolAnalysisManager::instance_->analysis_map_.contains(module)) {
    SymbolAnalysisManager::instance_->analysis_map_[module] =
        new SymbolAnalysis(module);
  }
  return SymbolAnalysisManager::instance_->analysis_map_[module];
}

void SymbolAnalysis::_insertInModule(LLHPatternRewriter* builder, Operation* op,
                                     Value value) const {
  ModuleOp module;
  if (llvm::isa<BlockArgument>(value)) {
    module = value.getParentBlock()->getParent()->getParentOfType<ModuleOp>();
  } else {
    module = value.getDefiningOp()->getParentOfType<ModuleOp>();
  }
  CHECK(llc::MLIR, module);
  auto& block = module->getRegion(0).getBlocks().front();
  op->remove();
  block.push_front(op);
}

void SymbolAnalysis::_insertToSymbolModule(LLHPatternRewriter* builder,
                                           Operation* op) const {
  auto symbol_module = getSymbolModule();
  ModuleOp module = llvm::cast_or_null<ModuleOp>(symbol_module_);
  CHECK(llc::MLIR, module);
  auto& block = module->getRegion(0).getBlocks().front();
  op->remove();
  block.push_front(op);
}

SymbolicIntOp SymbolAnalysis::buildNewSymbolFrom(Value value) {
  // std::lock_guard<std::mutex> lock(mutex_);
  auto module = value.getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module));
  std::string symbol = "s" + std::to_string(next_symbol_id_);
  next_symbol_id_++;
  while (symbols_table_.count(symbol)) {
    symbol = "s" + std::to_string(next_symbol_id_);
    next_symbol_id_++;
  }
  LLHPatternRewriter builder(value.getContext());
  auto symbol_op =
      builder.create<SymbolicIntOp>(builder.getUnknownLoc(), symbol);
  _insertInModule(&builder, symbol_op, value);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildSymbolFrom(Value value,
                                                   std::string symbol) {
  LLHPatternRewriter builder(value.getContext());
  if (symbols_table_.count(symbol)) {
    return llvm::cast<SymbolicIntOp>(symbols_table_[symbol]);
  }
  auto symbol_op =
      builder.create<SymbolicIntOp>(builder.getUnknownLoc(), symbol);
  _insertInModule(&builder, symbol_op, value);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
};

SymbolicIntOp SymbolAnalysis::getOrBuildConstSymbolFrom(Value value,
                                                        size_t val) {
  std::string symbol = "c" + std::to_string(val);
  return getOrBuildSymbolFrom(value, symbol);
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
      auto new_symbol = symbols_analysis->buildNewSymbolFrom(value);
      symbols.push_back(new_symbol.getSymName());
    } else {
      auto const_symbol =
          symbols_analysis->getOrBuildConstSymbolFrom(value, dim);
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
      CHECK(llc::SymbolInfer, symbols_table_.contains(symbols[i].str()));
      new_symbols.push_back(symbols[i]);
      continue;
    }
    auto dim = shape[i];
    if (dim == ShapedType::kDynamic) {
      auto new_symbol = symbols_analysis->buildNewSymbolFrom(value);
      new_symbols.push_back(new_symbol.getSymName());
    } else {
      auto const_symbol =
          symbols_analysis->getOrBuildConstSymbolFrom(value, dim);
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

bool SymbolAnalysis ::isExtraSymbolicInferOp(Operation* op) {
  return isa<DimOp, ConstantOp, DivOp, AddOp, MulOp, SubOp>(op);
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
  auto op = value.getDefiningOp();
  return getSymbolAttr(op);
}

llvm::StringRef SymbolAnalysis::getOrBuildSymbolAttr(Operation* op) {
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

llvm::StringRef SymbolAnalysis::getOrBuildSymbolAttr(Value value) {
  auto op = value.getDefiningOp();
  return getOrBuildSymbolAttr(op);
}

SymbolRelationOp SymbolAnalysis::buildSymbolRelation(
    Value symbol, Value relation, SymbolRelation relation_kind) {
  if (!hasSymbolAttr(symbol)) return nullptr;
  if (!hasSymbolAttr(relation)) return nullptr;
  auto symbol_name = getSymbolAttr(symbol);
  auto relation_name = getSymbolAttr(relation);
  LLHPatternRewriter builder(symbol.getContext());
  auto relation_op = builder.create<SymbolRelationOp>(
      builder.getUnknownLoc(), symbol_name, relation_name, relation_kind);
  _insertToSymbolModule(&builder, relation_op);
  UNIMPLEMENTED(llc::SymbolInfer) << "stored relation";
  return relation_op;
}

SymbolBinaryRelationOp SymbolAnalysis::buildSymbolRelation(
    Value symbol, Value relation_lhs, Value relation_rhs,
    SymbolRelation relation_kind) {
  if (!hasSymbolAttr(symbol)) return nullptr;
  if (!hasSymbolAttr(relation_lhs)) return nullptr;
  if (!hasSymbolAttr(relation_rhs)) return nullptr;
  auto symbol_name = getSymbolAttr(symbol);
  auto lhs_name = getSymbolAttr(relation_lhs);
  auto rhs_name = getSymbolAttr(relation_rhs);
  LLHPatternRewriter builder(symbol.getContext());
  auto relation_op = builder.create<SymbolBinaryRelationOp>(
      builder.getUnknownLoc(), symbol_name, lhs_name, rhs_name, relation_kind);
  _insertToSymbolModule(&builder, relation_op);
  UNIMPLEMENTED(llc::SymbolInfer) << "stored relation";
  return relation_op;
}

ModuleOp SymbolAnalysis::getSymbolModule() const {
  return cast<ModuleOp>(symbol_module_);
}

void SymbolAnalysis::debugPrintSymbols() {
  for (auto pair : symbols_table_) {
    pair.second->dump();
  }
}

}  // namespace mlir::llh
