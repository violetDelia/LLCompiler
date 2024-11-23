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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "Conversion/LLHToHLO/LLHToHLO.h"
#include "Dialect/LLH/IR/LLHOps.h"
#include "Dialect/Utility/RewritePattern.h"
#include "Dialect/Utility/Type.h"
#include "Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/StablehloOps.h"
namespace mlir {
#define GEN_PASS_DEF_LLHPREPROCESSINGFORHLOPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// legal func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct ReluOpSwitch : public LLHOpRewritePattern<ReluOp> {
  using LLHOpRewritePattern<ReluOp>::LLHOpRewritePattern;
  LogicalResult match(ReluOp op) const final { return llvm::success(); }
  void rewrite(ReluOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res.getType());
    auto res_ele_type = res_type.getElementType();
    DenseElementsAttr value;

    if (isa<IntegerType>(res_ele_type)) {
      value = SplatElementsAttr::get(RankedTensorType::get({1}, res_ele_type),
                                     IntegerAttr::get(res_ele_type, 0));
    } else if (isa<FloatType>(res_ele_type)) {
      value = SplatElementsAttr::get(RankedTensorType::get({1}, res_ele_type),
                                     FloatAttr::get(res_ele_type, 0));
    } else {
      UNIMPLEMENTED(llc::MLIR_PASS);
    }
    auto zore = rewriter.create<ConstantOp>(loc, value);
    rewriter.replaceOpWithNewOp<MaxOp>(op, TypeRange{res_type},
                                       ValueRange{input, zore},
                                       op->getAttrDictionary().getValue());
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateLLHPreprocessingForHLOPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<ReluOpSwitch>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
struct LLHPreprocessingForHLOPass
    : impl::LLHPreprocessingForHLOPassBase<LLHPreprocessingForHLOPass> {
  using impl::LLHPreprocessingForHLOPassBase<
      LLHPreprocessingForHLOPass>::LLHPreprocessingForHLOPassBase;
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void LLHPreprocessingForHLOPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateLLHPreprocessingForHLOPassPatterns(patterns);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
