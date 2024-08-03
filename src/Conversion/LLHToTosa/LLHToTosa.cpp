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
#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
namespace mlir {
#define GEN_PASS_DEF_CONVERTAFFINETOSTANDARD
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

using namespace mlir;
using namespace mlir::llh;

//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//**
namespace {
struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult match(ReluOp op) const final { return success(); }
  void rewrite(ReluOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {}
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::llh::populateLLHToTosaConversionPatterns(
    TypeConverter& converter, RewritePatternSet& patterns) {}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void mlir::llh::configLLHToTosaConversionTarget(ConversionTarget& target) {}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void mlir::llh::initLLHtoTosaConversionTypeConverter(TypeConverter& converter) {
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//
namespace {
struct LLHToTosaConversion final
    : impl::LLHToTosaConversionPassBase<LLHToTosaConversion> {
  using impl::LLHToTosaConversionPassBase<
      LLHToTosaConversion>::LLHToTosaConversionPassBase;

  void runOnOperation() override final {
    ConversionTarget target(getContext());
    configLLHToTosaConversionTarget(target);
    TypeConverter converter;
    initLLHtoTosaConversionTypeConverter(converter);
    RewritePatternSet patterns(&getContext());
    populateLLHToTosaConversionPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  };
};
}  // namespace
