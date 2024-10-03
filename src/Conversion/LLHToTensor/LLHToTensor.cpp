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
#include "llcompiler/Conversion/LLHToTensor/LLHToTensor.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
namespace mlir {
#define GEN_PASS_DEF_CONVERTLLHTOTENSORPASS
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
namespace {

//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct DimOpLowing : public OpConversionPattern<DimOp> {
  using OpConversionPattern<DimOp>::OpConversionPattern;
  LogicalResult match(DimOp op) const final { return llvm::success(); }

  void rewrite(DimOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateConvertLLHToTensorPassPatterns(TypeConverter& converter,
                                            RewritePatternSet& patterns) {
  auto context = patterns.getContext();
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configConvertLLHToTensorPassTarget(ConversionTarget& target) {
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void initConvertLLHToTensorPassTypeConverter(TypeConverter& converter) {
  auto shaped_repalce = [](ShapedType type) { return type; };
  auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
  converter.addConversion(ranked_tensor_replace);
  converter.addConversion(shaped_repalce);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
struct ConvertLLHToTensorPass
    : impl::ConvertLLHToTensorPassBase<ConvertLLHToTensorPass> {
  using impl::ConvertLLHToTensorPassBase<
      ConvertLLHToTensorPass>::ConvertLLHToTensorPassBase;
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void ConvertLLHToTensorPass::runOnOperation() {
  LLC_RUN_IN_PASS
  ConversionTarget target(getContext());
  configConvertLLHToTensorPassTarget(target);
  TypeConverter converter;
  initConvertLLHToTensorPassTypeConverter(converter);
  RewritePatternSet patterns(&getContext());
  populateConvertLLHToTensorPassPatterns(converter, patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
