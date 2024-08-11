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
#include <cstdint>
#include <cstdio>

#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/Macro.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
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
bool check_matmal_illegal(Operation* op) {
  auto matmal = cast_or_null<MatMulOp>(op);
  if (!matmal) return false;
  auto left_type = cast_or_null<ShapedType>(matmal.getLhs().getType());
  auto right_type = cast_or_null<ShapedType>(matmal.getRhs().getType());
  auto result_type = cast_or_null<ShapedType>(matmal.getResult().getType());
  if (!left_type || !right_type || !result_type) return false;
  if (left_type.getRank() != left_type.getRank()) return false;
  if (left_type.getRank() != result_type.getRank()) return false;
  if (left_type.getRank() != 2 || left_type.getRank() != 3) return false;
  return true;
}

bool check_conv_illegal(Operation* op) {
  auto conv = cast_or_null<ConvOp>(op);
  if (!conv) return false;
  if (conv.getGroup() != 1) return false;
  auto x_type = cast_or_null<ShapedType>(conv.getX().getType());
  auto w_type = cast_or_null<ShapedType>(conv.getW().getType());
  if (x_type || w_type) return false;
  if (x_type.getRank() != w_type.getRank()) return false;
  if (x_type.getRank() != 4 || w_type.getRank() != 5) return false;
  return true;
}

namespace {};
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
    auto out_shape = cast<ShapedType>(out).getShape();
    auto out_ele_type = cast<ShapedType>(out).getElementType();
    auto attrs = adaptor.getAttributes().getValue();
    auto const_op =
        llc::create_tosa_const(&rewriter, out_shape, {0}, out_ele_type, loc);
    auto new_op = rewriter.create<mlir::tosa::MaximumOp>(
        loc, ::mlir::TypeRange{out},
        ::mlir::ValueRange{input, const_op->getResult(0)}, attrs);
    rewriter.replaceOp(op, new_op);
  }
};

struct WeightOpLowering : public OpConversionPattern<WeightOp> {
  using OpConversionPattern<WeightOp>::OpConversionPattern;
  LogicalResult match(WeightOp op) const final { return success(); }
  void rewrite(WeightOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto out = op.getResult().getType();
    auto attrs = adaptor.getAttributes().getValue();
    auto types = ::mlir::TypeRange{out};
    auto new_op = rewriter.create<mlir::tosa::ConstOp>(
        loc, types, ::mlir::ValueRange{}, attrs);
    new_op.setValueAttr(adaptor.getValueAttr());
    llc::add_is_weight_attr(new_op, true);
    rewriter.replaceOp(op, new_op);
  }
};

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;
  LogicalResult match(ConstantOp op) const final { return success(); }
  void rewrite(ConstantOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto out = op.getResult().getType();
    auto attrs = adaptor.getAttributes().getValue();
    auto types = ::mlir::TypeRange{out};
    auto new_op = rewriter.create<mlir::tosa::ConstOp>(
        loc, types, ::mlir::ValueRange{}, attrs);
    new_op.setValueAttr(adaptor.getValueAttr());
    rewriter.replaceOp(op, new_op);
  }
};

struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;
  LogicalResult match(MatMulOp op) const final { return success(); }
  void rewrite(MatMulOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto out = op.getResult();
    auto out_type = cast<ShapedType>(out.getType());
    auto attrs = adaptor.getAttributes().getValue();
    if (out_type.getRank() == 3) {
      auto new_op = rewriter.create<tosa::MatMulOp>(
          loc, ::mlir::TypeRange{out_type}, adaptor.getOperands(), attrs);
      rewriter.replaceOp(op, new_op);
    }
    if (out_type.getRank() == 2) {
      auto left = adaptor.getLhs();
      auto left_shape = llc::get_shape_form(left.getType());
      left_shape.insert(left_shape.begin(), 1);
      auto left_reshape_op =
          rewriter.create<mlir::tosa::ReshapeOp>(loc, left, left_shape);
      left_reshape_op.dump();
      auto right = adaptor.getRhs();
      auto right_shape = llc::get_shape_form(right.getType());
      right_shape.insert(right_shape.begin(), 1);
      auto right_reshape_op =
          rewriter.create<mlir::tosa::ReshapeOp>(loc, right, right_shape);
      right_reshape_op.dump();
      auto new_out_shape = llc::get_shape_form(out_type);
      new_out_shape.insert(new_out_shape.begin(), 1);
      auto new_out_type =
          RankedTensorType::get(new_out_shape, out_type.getElementType());
      auto matmul_op = rewriter.create<tosa::MatMulOp>(
          loc, ::mlir::TypeRange{new_out_type},
          ::mlir::ValueRange{left_reshape_op.getResult(),
                             right_reshape_op.getResult()},
          attrs);
      auto reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
          loc, matmul_op.getResult(), out_type.getShape());
      rewriter.replaceOp(op, reshape_op);
    }
  }
};

struct ConvOpLowering : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;
  LogicalResult match(ConvOp op) const final { return success(); }
  void rewrite(ConvOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto out = op.getResult();
    auto out_type = cast<ShapedType>(out.getType());
    auto atrrs = op->getAttrDictionary().getValue();
    Operation* new_op;
    if (out_type.getRank() == 4) {
      new_op = rewriter.create<tosa::Conv2DOp>(loc, ::mlir::TypeRange{out},
                                               adaptor.getOperands(), atrrs);
    }
    if (out_type.getRank() == 5) {
      new_op = rewriter.create<tosa::Conv3DOp>(loc, ::mlir::TypeRange{out},
                                               adaptor.getOperands(), atrrs);
    }
    rewriter.replaceOp(op, new_op);
  };
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::llh::populateLLHToTosaConversionPatterns(
    TypeConverter& converter, RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<ReluOpLowering>(converter, context);
  patterns.add<WeightOpLowering>(converter, context);
  patterns.add<ConstantOpLowering>(converter, context);
  patterns.add<MatMulOpLowering>(converter, context);
  patterns.add<ConvOpLowering>(converter, context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void mlir::llh::configLLHToTosaConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<ConstantOp>();
  target.addIllegalOp<WeightOp>();
  // target.addIllegalOp<ReluOp>();
  target.addDynamicallyLegalOp<MatMulOp>(check_matmal_illegal);
  target.addDynamicallyLegalOp<ConvOp>(check_conv_illegal);
  target.addLegalDialect<mlir::tosa::TosaDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void mlir::llh::initLLHtoTosaConversionTypeConverter(TypeConverter& converter) {
  auto shaped_repalce = [](ShapedType type) { return type; };
  auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
  converter.addConversion(ranked_tensor_replace);
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
