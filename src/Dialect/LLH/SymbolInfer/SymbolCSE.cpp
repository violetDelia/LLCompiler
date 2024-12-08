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

#include <cstdint>
#include <map>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_SYMBOLCSEPASS
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
int getKeyOf(Type type) {
  if (isa<IndexType>(type)) return 0;
  if (isa<IntegerType>(type)) return 1;
  if (isa<FloatType>(type)) return 2;
  UNIMPLEMENTED(llc::SymbolInfer);
  return -1;
}

void foldSymbol(func::FuncOp func) {
  std::map<int, std::map<std::string, mlir::Value>> symbol_map;
  auto builder = LLHPatternRewriter(func);
  auto init_map = [&symbol_map](Operation* op) {
    if (!SymbolAnalysis::isExtraSymbolAttrInferOp(op)) return;
    auto maybe_symbol_attr = op->getAttr(llc::SymbolIntAttr);
    if (!maybe_symbol_attr) return;
    auto symbol_attr =
        llvm::cast_or_null<mlir::FlatSymbolRefAttr>(maybe_symbol_attr);
    if (!symbol_attr) return;
    auto symbol = symbol_attr.getValue();
    auto key = getKeyOf(op->getResult(0).getType());
    if (symbol_map[key].contains(symbol.str())) return;
    symbol_map[key][symbol.str()] = op->getResult(0);
  };
  auto replace_symbol_value = [&symbol_map, &builder](Operation* op) {
    if (!SymbolAnalysis::isExtraSymbolAttrInferOp(op)) return;
    auto maybe_symbol_attr = op->getAttr(llc::SymbolIntAttr);
    if (!maybe_symbol_attr) return;
    auto symbol_attr =
        llvm::cast_or_null<mlir::FlatSymbolRefAttr>(maybe_symbol_attr);
    if (!symbol_attr) return;
    auto symbol = symbol_attr.getValue();
    auto key = getKeyOf(op->getResult(0).getType());
    auto front_symbol_op = symbol_map[key][symbol.str()].getDefiningOp();
    if (front_symbol_op == op) return;

    builder.replaceOp( op,front_symbol_op);
  };
  func->walk(init_map);
  func->walk(replace_symbol_value);
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateSymbolCSEPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  // populateWithGenerated(patterns);
}
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct SymbolCSEPass : llh::impl::SymbolCSEPassBase<SymbolCSEPass> {
  void runOnOperation() override;
};

}  // namespace
void SymbolCSEPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  auto fold_symbol_dim = [](func::FuncOp func) { foldSymbol(func); };
  module->walk(fold_symbol_dim);
  RewritePatternSet patterns(context);
  populateSymbolCSEPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
