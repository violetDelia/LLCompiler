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

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/IRExtension/IR/Enums.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/Macro.h"
#include "llcompiler/Dialect/Utility/Tool.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
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

#define BUILD_ATTR(judge, Ty, shape)                        \
  if (judge) {                                              \
    llvm::ArrayRef<Ty> value(0);                            \
    auto attr = mlir::DenseElementsAttr::get(shape, value); \
    return attr;                                            \
  }

DenseElementsAttr genZoreElementAttr(Value value) {
  auto val = cast<mlir::RankedTensorType>(value.getType());
  val.getRank();
  CHECK(llc::MLIR, val);
  auto tensor = llc::cloneTensorWithEncoding(val, mlir::ex::Layout::Any);
  auto type = tensor.getElementType();
  BUILD_ATTR(type.isInteger(1), bool, tensor)
  BUILD_ATTR(type.isSignedInteger(8), int8_t, tensor)
  BUILD_ATTR(type.isSignedInteger(16), int16_t, tensor)
  BUILD_ATTR(type.isSignedInteger(32), int32_t, tensor)
  BUILD_ATTR(type.isSignedInteger(64), int64_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(8), uint8_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(16), uint16_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(32), uint32_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(64), uint64_t, tensor)
  BUILD_ATTR(type.isF32(), float, tensor)
  BUILD_ATTR(type.isF64(), double, tensor)
  UNIMPLEMENTED(llc::MLIR);
  return {};
}

RankedTensorType cloneTensorWithLayoutAny(Value value) {
  auto val = cast<mlir::RankedTensorType>(value.getType());
  CHECK(llc::MLIR, val);
  return llc::cloneTensorWithEncoding(val, mlir::ex::Layout::Any);
}

mlir::DenseI64ArrayAttr UnsqueezeShape(Value value, int dim = 0) {
  auto shape = cast<mlir::ShapedType>(value.getType());
  CHECK(llc::MLIR, shape);
  return DenseI64ArrayAttr::get(value.getContext(),
                                llc::getUnsqueezeShape(shape, dim));
}

mlir::DenseI64ArrayAttr GetShape(Value value) {
  auto shape = cast<mlir::ShapedType>(value.getType());
  CHECK(llc::MLIR, shape);
  return DenseI64ArrayAttr::get(value.getContext(), shape.getShape());
}

mlir::RankedTensorType UnsqueezeTensor(Value value, int dim = 0) {
  auto tensor = cast<mlir::RankedTensorType>(value.getType());
  CHECK(llc::MLIR, tensor);
  return llc::getUnsqueezeTensor(tensor);
}

mlir::RankedTensorType SqueezeTensor(Value value, int dim = 0) {
  auto tensor = cast<mlir::RankedTensorType>(value.getType());
  CHECK(llc::MLIR, tensor);
  return llc::getSqueezeTensor(tensor);
}

#undef BUILD_ATTR
//===----------------------------------------------------------------------===//
// illegal func
//===----------------------------------------------------------------------===//
bool check_matmal_illegal(Operation* op) {
  auto matmal = cast_or_null<MatMulOp>(op);
  if (!matmal) return false;
  auto left_type = cast_or_null<RankedTensorType>(matmal.getLhs().getType());
  auto right_type = cast_or_null<RankedTensorType>(matmal.getRhs().getType());
  auto result_type =
      cast_or_null<RankedTensorType>(matmal.getResult().getType());
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
  auto x_type = cast_or_null<RankedTensorType>(conv.getX().getType());
  auto w_type = cast_or_null<RankedTensorType>(conv.getW().getType());
  if (x_type || w_type) return false;
  if (x_type.getRank() != w_type.getRank()) return false;
  if (x_type.getRank() != 4 || w_type.getRank() != 5) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
namespace {
#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.inc"

struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult match(ReluOp op) const final { return success(); }
  void rewrite(ReluOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    LLC_RUN_IN_PATTERN
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
    LLC_RUN_OUT_PATTERN
  }
};

struct WeightOpLowering : public OpConversionPattern<WeightOp> {
  using OpConversionPattern<WeightOp>::OpConversionPattern;
  LogicalResult match(WeightOp op) const final { return success(); }
  void rewrite(WeightOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    LLC_RUN_IN_PATTERN
    auto loc = op.getLoc();
    auto out = op.getResult().getType();
    auto attrs = adaptor.getAttributes().getValue();
    auto types = ::mlir::TypeRange{out};
    auto new_op = rewriter.create<mlir::tosa::ConstOp>(
        loc, types, ::mlir::ValueRange{}, attrs);
    new_op.setValueAttr(adaptor.getValueAttr());
    rewriter.replaceOp(op, new_op);
    LLC_RUN_OUT_PATTERN
  }
};

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;
  LogicalResult match(ConstantOp op) const final { return success(); }
  void rewrite(ConstantOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    LLC_RUN_IN_PATTERN
    auto loc = op.getLoc();
    auto out = op.getResult().getType();
    auto attrs = adaptor.getAttributes().getValue();
    auto types = ::mlir::TypeRange{out};
    auto new_op = rewriter.create<mlir::tosa::ConstOp>(
        loc, types, ::mlir::ValueRange{}, attrs);
    new_op.setValueAttr(adaptor.getValueAttr());
    rewriter.replaceOp(op, new_op);
    LLC_RUN_OUT_PATTERN
  }
};

struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;
  LogicalResult match(MatMulOp op) const final { return success(); }
  void rewrite(MatMulOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    LLC_RUN_IN_PATTERN
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
      auto left_reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
          loc, UnsqueezeTensor(left), left, UnsqueezeShape(left));
      auto right = adaptor.getRhs();
      auto right_reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
          loc, UnsqueezeTensor(right), right, UnsqueezeShape(right));
      auto new_out_shape = llc::getShapeFrom(out_type);
      new_out_shape.insert(new_out_shape.begin(), 1);
      auto new_out_type =
          RankedTensorType::get(new_out_shape, out_type.getElementType());
      auto matmul_op = rewriter.create<tosa::MatMulOp>(
          loc, ::mlir::TypeRange{UnsqueezeTensor(out)},
          ::mlir::ValueRange{left_reshape_op, right_reshape_op}, attrs);
      auto reshape_op = rewriter.create<mlir::tosa::ReshapeOp>(
          loc, out_type, matmul_op, GetShape(out));
      rewriter.replaceOp(op, reshape_op);
    }
    LLC_RUN_OUT_PATTERN
  }
};

struct ConvOpLowering : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;
  LogicalResult match(ConvOp op) const final { return success(); }
  void rewrite(ConvOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    LLC_RUN_IN_PATTERN
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
    LLC_RUN_OUT_PATTERN
  };
};

// struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
//   using OpConversionPattern<TransposeOp>::OpConversionPattern;
//   LogicalResult match(TransposeOp op) const final { return success(); }
//   void rewrite(TransposeOp op, OpAdaptor adaptor,
//                ConversionPatternRewriter& rewriter) const final {
//     LLC_RUN_IN_PATTERN
//     auto loc = op.getLoc();
//     auto out = op.getResult();
//     auto input = op.getInput();
//     auto perms = op.getPermsAttr();
//     auto atrrs = op->getAttrDictionary().getValue();
//     auto const_return = genTensorFromAttr(perms);
//     auto const_op = rewriter.create<tosa::ConstOp>(loc, const_return, perms);
//     auto con = const_op.getResult();
//     auto new_op = rewriter.create<tosa::TransposeOp>(
//         loc, ::mlir::TypeRange{out.getType()},
//         ::mlir::ValueRange{op.getInput(), const_op->getResult(0)}, atrrs);
//     rewriter.replaceOp(op, new_op);
//     LLC_RUN_OUT_PATTERN
//   };
// };
}  // namespace

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateLLHToTosaConversionPatterns(TypeConverter& converter,
                                         RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<ReluOpLowering>(converter, context);
   patterns.add<WeightOpLowering>(converter, context);
   patterns.add<ConstantOpLowering>(converter, context);
  patterns.add<MatMulOpLowering>(converter, context);
  patterns.add<ConvOpLowering>(converter, context);
  populateWithGenerated(patterns);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configLLHToTosaConversionTarget(ConversionTarget& target) {
  target.addIllegalOp<ConstantOp>();
  target.addIllegalOp<WeightOp>();
  target.addIllegalOp<ReluOp>();
  target.addDynamicallyLegalOp<MatMulOp>(check_matmal_illegal);
  target.addDynamicallyLegalOp<ConvOp>(check_conv_illegal);
  target.addLegalDialect<mlir::tosa::TosaDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void initLLHtoTosaConversionTypeConverter(TypeConverter& converter) {
  auto shaped_repalce = [](ShapedType type) { return type; };
  auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
  converter.addConversion(ranked_tensor_replace);
  converter.addConversion(shaped_repalce);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct LLHToTosaConversion : impl::ConvertLLHToTosaBase<LLHToTosaConversion> {
  using impl::ConvertLLHToTosaBase<LLHToTosaConversion>::ConvertLLHToTosaBase;
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void LLHToTosaConversion::runOnOperation() {
  LLC_RUN_IN_PASS
  ConversionTarget target(getContext());
  configLLHToTosaConversionTarget(target);
  TypeConverter converter;
  initLLHtoTosaConversionTypeConverter(converter);
  RewritePatternSet patterns(&getContext());
  populateLLHToTosaConversionPatterns(converter, patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
