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

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_REMOVESYMBOLPASS
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
template <class Op>
struct RemoveOp : public LLHOpRewritePattern<Op> {
  using LLHOpRewritePattern<Op>::LLHOpRewritePattern;
  LogicalResult match(Op op) const final { return llvm::success(); }
  void rewrite(Op op, LLHPatternRewriter& rewriter) const final {
    rewriter.eraseOp(op);
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateRemoveSymbolPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<RemoveOp<SymbolicIntOp>>(context);
  patterns.add<RemoveOp<SymbolicCastOp>>(context);
  patterns.add<RemoveOp<SymbolBindOp>>(context);
  patterns.add<RemoveOp<SymbolRelationMapOp>>(context);
  patterns.add<RemoveOp<SymbolRelationOp>>(context);
  patterns.add<RemoveOp<SymbolBinaryRelationOp>>(context);
  patterns.add<RemoveOp<EncodingBindOp>>(context);
}
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct RemoveSymbolPass : llh::impl::RemoveSymbolPassBase<RemoveSymbolPass> {
  void runOnOperation() override;
};

}  // namespace
void RemoveSymbolPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  populateRemoveSymbolPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
  auto analysis = SymbolAnalysis::getInstance(module);
  analysis->cleanCache();
  LLC_RUN_OUT_PASS
}
