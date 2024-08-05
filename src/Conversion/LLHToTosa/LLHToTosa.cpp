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
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
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
    CHECK(llc::LLHTOTOSA, isa<ShapedType>(out)) << "Unexpected result type";
    auto const_op =
        llc::create_tosa_const(&rewriter, cast<ShapedType>(out).getShape(), {0},
                               cast<ShapedType>(out).getElementType(), loc);
    auto new_op = rewriter.create<mlir::tosa::MaximumOp>(
        loc, ::mlir::TypeRange{op.getResult().getType()},
        ::mlir::ValueRange{input, const_op->getResult(0)},
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
  // target.addLegalDialect<mlir::llh::LLHDialect>();
  target.addLegalDialect<mlir::tosa::TosaDialect>();
}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void mlir::llh::initLLHtoTosaConversionTypeConverter(TypeConverter& converter) {
  auto shaped_repalce = [](ShapedType type) { return type; };
  // auto rank_tensor_replace = [](RankedTensorType type) { return type; };
  //  converter.addConversion(shape_repalce);
  converter.addConversion(shaped_repalce);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//
namespace {
struct LLHToTosaConversion : impl::ConvertLLHToTosaBase<LLHToTosaConversion> {
  using impl::ConvertLLHToTosaBase<LLHToTosaConversion>::ConvertLLHToTosaBase;

  void runOnOperation() override {
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
