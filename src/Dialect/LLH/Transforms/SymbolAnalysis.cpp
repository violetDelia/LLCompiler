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

#include "llcompiler/Dialect/LLH/Transforms/SymbolAnalysis.h"

#include <cstdint>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::llh {

SymbolAnalysis* SymbolAnalysis::instance_ = new (std::nothrow) SymbolAnalysis;

SymbolAnalysis::SymbolAnalysis() {}

SymbolAnalysis::~SymbolAnalysis() {}

SymbolAnalysis* SymbolAnalysis::getInstance() { return instance_; }

void SymbolAnalysis::deleteInstance() {
  if (instance_) {
    delete instance_;
    instance_ = NULL;
  }
}

void SymbolAnalysis::_insertOp(RewriterBase* builder, Operation* op,
                               Operation* base) const {
  auto module = op->getParentOfType<ModuleOp>();
  CHECK(llc::MLIR, module);
  auto& block = module->getRegion(0).getBlocks().front();
  op->remove();
  block.push_front(op);
};

SymbolicIntOp SymbolAnalysis::buildNewSymbol(RewriterBase* builder,
                                             Operation* base) {
  // std::lock_guard<std::mutex> lock(mutex_);
  std::string symbol = "s" + std::to_string(next_symbol_id_);
  next_symbol_id_++;
  auto symbol_op =
      builder->create<SymbolicIntOp>(builder->getUnknownLoc(), symbol);
  _insertOp(builder, symbol_op, base);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
}

SymbolicIntOp SymbolAnalysis::getOrBuildConstSymbol(RewriterBase* builder,
                                                    Operation* base, int val) {
  std::string symbol = "c" + std::to_string(val);
  if (symbols_table_.count(symbol)) {
    return llvm::cast<SymbolicIntOp>(symbols_table_[symbol]);
  }
  auto symbol_op =
      builder->create<SymbolicIntOp>(builder->getUnknownLoc(), symbol);
  _insertOp(builder, symbol_op, base);
  symbols_table_[symbol_op.getSymName().str()] = symbol_op;
  return symbol_op;
};

SymbolRelationsOp SymbolAnalysis::buildRelations(
    RewriterBase* builder, Operation* base, llvm::StringRef symbol,
    llvm::ArrayRef<llvm::StringRef> relations, AffineExpr expr) {
  auto affin_map = AffineMap::get(1, relations.size(), expr);
  auto relations_op = builder->create<SymbolRelationsOp>(
      builder->getUnknownLoc(),
      SymbolRefAttr::get(builder->getStringAttr(symbol)),
      builder->getStrArrayAttr(relations), AffineMapAttr::get(affin_map));
      relations_op->dump();
  _insertOp(builder, relations_op, base);
  return relations_op;
}

void SymbolAnalysis::debugPrintSymbols() {
  for (auto pair : symbols_table_) {
    DINFO << pair.first;
    pair.second->dump();
  }
}

}  // namespace mlir::llh
