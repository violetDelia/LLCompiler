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
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLHTOTOSA
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
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto out = op.getResult().getType();
    rewriter.setInsertionPointAfter(op);
    auto ops =
        llc::expand_const_to(&rewriter, 0, out, cast<RankedTensorType>(out));
    for (auto op : ops) {
      rewriter.insert(op);
    }
    auto new_op = rewriter.create<mlir::tosa::MaximumOp>(
        loc, ::mlir::TypeRange{op.getResult().getType()},
        ::mlir::ValueRange{input, ops[1]->getResult(0)},
        adaptor.getAttributes().getValue());
    rewriter.replaceOp(op, new_op);
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::llh::populateLLHToTosaConversionPatterns(
    TypeConverter& converter, RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<ReluOpLowering>(converter, context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void mlir::llh::configLLHToTosaConversionTarget(ConversionTarget& target) {
  target.addIllegalDialect<mlir::llh::LLHDialect>();
}

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
    : impl::ConvertLLHToTosaBase<LLHToTosaConversion> {
  using impl::ConvertLLHToTosaBase<LLHToTosaConversion>::ConvertLLHToTosaBase;

  void runOnOperation() override final {
    ConversionTarget target(getContext());
    configLLHToTosaConversionTarget(target);
    TypeConverter converter;
    initLLHtoTosaConversionTypeConverter(converter);
    RewritePatternSet patterns(&getContext());
    populateLLHToTosaConversionPatterns(converter, patterns);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  };
};
}  // namespace
