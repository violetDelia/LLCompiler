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

#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Transforms/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_SYMBOLCANONICALIZATION
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
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
struct replaceTorchSymbolicIntOp
    : public LLCOpRewritePattern<TorchSymbolicIntOp> {
  using LLCOpRewritePattern::LLCOpRewritePattern;
  LogicalResult match(TorchSymbolicIntOp op) const final {
    if (op->hasAttr(llc::SymbolGeneratedAttr)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(TorchSymbolicIntOp op,
               LLCPatternRewriter& rewriter) const final {
    auto module = op->getParentOfType<ModuleOp>();
    auto main_func = module.lookupSymbol("main");
    auto symbol_analysis = SymbolAnalysis::getInstance();
    auto symbol = symbol_analysis->buildSymbolInt(&rewriter, op);
    rewriter.moveOpBefore(symbol, main_func);
    llc::add_symbol_generate_attr(op);
  }
};

struct replaceSymbolicBindOp : public LLCOpRewritePattern<SymbolicBindOp> {
  using LLCOpRewritePattern::LLCOpRewritePattern;
  LogicalResult match(SymbolicBindOp op) const final {
    if (op->hasAttr(llc::StopRun)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(SymbolicBindOp op, LLCPatternRewriter& rewriter) const final {
    auto operand = op.getOperand();
    auto bind_shape = op.getBindSymbols();
    auto bind_type = llvm::cast_or_null<RankedTensorType>(operand.getType());
    CHECK(llc::MLIR_PASS, bind_type);
    bind_type.dump();
    auto rank = bind_type.getRank();
    auto exps = op.getExpressions();
    exps.dump();
    DINFO << exps.getNumDims();
    DINFO << exps.getNumInputs();
    DINFO << exps.getNumResults();
    DINFO << exps.getNumSymbols();
    auto x0 = exps.getResult(0);
    if(auto ccc =  cast<AffineDimExpr>(x0)) {
      DINFO << "dim"<<ccc.getPosition();;
      x0.dump();
    }
    auto dim = exps.getResultPosition(x0);
    if(dim.has_value())DINFO<<dim.value();
    // s.dump();
    DINFO << static_cast<int>(x0.getKind());

    // bind_shape.dump();
    op->dump();
    llc::add_stop_run_attr(op);
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateSymbolCanonicalizationPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<replaceTorchSymbolicIntOp>(context);
  patterns.add<replaceSymbolicBindOp>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct SymbolCanonicalizationPass
    : llh::impl::SymbolCanonicalizationBase<SymbolCanonicalizationPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void SymbolCanonicalizationPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto symbol_analysis = ::mlir::llh::SymbolAnalysis::getInstance();
  auto op = getOperation();
  auto* context = &getContext();
  RewritePatternSet patterns(context);
  populateSymbolCanonicalizationPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
  symbol_analysis->debugPrintSymbols();
  LLC_RUN_OUT_PASS
}

//===----------------------------------------------------------------------===//
// pass create
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> mlir::llh::createGenerateSymbolPass() {
  return std::make_unique<SymbolCanonicalizationPass>();
}
