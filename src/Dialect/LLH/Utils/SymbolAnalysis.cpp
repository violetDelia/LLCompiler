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
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
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

SymbolAnalysisManager* SymbolAnalysisManager::instance_ =
    new (std::nothrow) SymbolAnalysisManager;

std::mutex SymbolAnalysisManager::mutex_;

std::mutex SymbolAnalysis::mutex_;

StringRef SymbolAnalysis::UNKOW_SYMBOL = "UNKOW";

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

void SymbolAnalysis::_insertOp(RewriterBase* builder, Operation* op,
                               Value& value) const {
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

SymbolicIntOp SymbolAnalysis::buildNewSymbolFrom(Value& value) {
  // std::lock_guard<std::mutex> lock(mutex_);
  auto module = value.getParentRegion()->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, llvm::isa<ModuleOp>(module));
  std::string symbol = "s" + std::to_string(next_symbol_id_);
  next_symbol_id_++;
  while (symbols_table_.count(symbol)) {
    symbol = "s" + std::to_string(next_symbol_id_);
    next_symbol_id_++;
  }
  IRRewriter builder(value.getContext());
  auto symbol_op =
      builder.create<SymbolicIntOp>(builder.getUnknownLoc(), symbol);
  _insertOp(&builder, symbol_op, value);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildConstSymbolFrom(Value& value,
                                                        size_t val) {
  std::string symbol = "c" + std::to_string(val);
  IRRewriter builder(value.getContext());
  if (symbols_table_.count(symbol)) {
    return llvm::cast<SymbolicIntOp>(symbols_table_[symbol]);
  }
  auto symbol_op =
      builder.create<SymbolicIntOp>(builder.getUnknownLoc(), symbol);
  _insertOp(&builder, symbol_op, value);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
}

SymbolRelationsOp SymbolAnalysis::buildRelations(
    RewriterBase* builder, llvm::StringRef symbol,
    llvm::ArrayRef<llvm::StringRef> relations, AffineExpr expr) {
  UNIMPLEMENTED(llc::MLIR);
}

Value& SymbolAnalysis::addEncoding(Value& value, size_t result_pos) {
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

Value& SymbolAnalysis::addEncoding(Value& value,
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

void SymbolAnalysis::debugPrintSymbols() {
  for (auto pair : symbols_table_) {
    DINFO << pair.first;
    pair.second->dump();
  }
}

}  // namespace mlir::llh
