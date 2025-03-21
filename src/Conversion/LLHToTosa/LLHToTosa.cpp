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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Tool.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llcompiler/Support/MlirUtility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#define GEN_PASS_DEF_CONVERTLLHTOTOSAPASS
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
namespace {

RankedTensorType cloneTensorWithLayoutAny(Value value) {
  auto val = cast<mlir::RankedTensorType>(value.getType());
  CHECK(llc::MLIR, val);
  return val;
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
// legal func
//===----------------------------------------------------------------------===//
bool check_const_legal(Operation* op) {
  auto const_op = llvm::cast_or_null<ConstantOp>(op);
  if (!const_op) return false;
  auto type = const_op.getResult().getType();
  return !isa<RankedTensorType>(type);
}

bool check_matmal_legal(Operation* op) {
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
  DEBUG(llc::MLIR) << "??";
  return true;
}

//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult match(ReluOp op) const final { return success(); }
  void rewrite(ReluOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto input = adaptor.getInput();
    auto out = op.getResult().getType();
    auto out_shape = cast<ShapedType>(out).getShape();
    auto out_ele_type = cast<ShapedType>(out).getElementType();
    auto attrs = op->getAttrs();
    auto const_op =
        llc::create_tosa_const(&rewriter, out_shape, {0}, out_ele_type, loc);
    auto new_op = rewriter.create<mlir::tosa::MaximumOp>(
        loc, ::mlir::TypeRange{out},
        ::mlir::ValueRange{input, const_op->getResult(0)}, attrs);
    rewriter.replaceOp(op, new_op);
  }
};

// struct WeightOpLowering : public OpConversionPattern<WeightOp> {
//   using OpConversionPattern<WeightOp>::OpConversionPattern;
//   LogicalResult match(WeightOp op) const final { return success(); }
//   void rewrite(WeightOp op, OpAdaptor adaptor,
//                ConversionPatternRewriter& rewriter) const final {
//
//     Loc_And_Context;
//     auto out = op.getResult().getType();
//     auto attrs = op->getAttrs();
//     auto types = ::mlir::TypeRange{out};
//     auto new_op = rewriter.create<mlir::tosa::ConstOp>(
//         loc, types, ::mlir::ValueRange{}, attrs);
//     new_op.setValueAttr(adaptor.getValueAttr());
//     llc::add_is_weight_attr(new_op, true);
//     rewriter.replaceOp(op, new_op);
//
//   }
// };

struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;
  LogicalResult match(MatMulOp op) const final { return success(); }
  void rewrite(MatMulOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto out = op.getResult();
    auto out_type = cast<ShapedType>(out.getType());
    auto attrs = op->getAttrs();
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
  }
};

struct LLHConvOpToTosa : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;
  LogicalResult match(ConvOp op) const final { return success(); }
  void rewrite(ConvOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto res = op.getResult();
    auto res_type = cast<ShapedType>(res.getType());
    auto res_ele_type = res_type.getElementType();
    auto atrrs = op->getAttrDictionary().getValue();
    auto input = op.getX();
    auto weight = op.getW();
    auto bias = llc::create_tosa_const(&rewriter, {1}, {0}, res_ele_type, loc);
    auto dilation = op.getDilationAttr();
    auto pad = op.getPadAttr();
    auto stride = op.getStrideAttr();

    if (res_type.getRank() == 4) {
      rewriter.replaceOpWithNewOp<tosa::Conv2DOp>(op, res_type, input, weight,
                                                  bias, pad, stride, dilation,
                                                  TypeAttr::get(res_ele_type));
    }
    if (res_type.getRank() == 5) {
      rewriter.replaceOpWithNewOp<tosa::Conv3DOp>(op, res_type, input, weight,
                                                  bias, pad, stride, dilation,
                                                  TypeAttr::get(res_ele_type));
    }
  };
};

struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern<TransposeOp>::OpConversionPattern;
  LogicalResult match(TransposeOp op) const final { return success(); }
  void rewrite(TransposeOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto out = op.getResult();
    auto input = op.getInput();
    auto perms = op.getPermsAttr();
    auto atrrs = op->getAttrs();
    auto const_shape = SmallVector<int64_t>(1, perms.size());
    auto const_out = RankedTensorType::get(const_shape, rewriter.getI64Type());
    auto const_value = llc::genDenseElementsFromArrayAttr(perms);
    auto const_op = rewriter.create<tosa::ConstOp>(loc, const_out, const_value);
    auto new_op = rewriter.create<tosa::TransposeOp>(
        loc, ::mlir::TypeRange{out.getType()},
        ::mlir::ValueRange{input, const_op}, atrrs);
    rewriter.replaceOp(op, new_op);
  };
};

struct LLHMulOpToTosa : public OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  LogicalResult match(MulOp op) const final { return success(); }
  void rewrite(MulOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto types = op->getResultTypes();
    auto operands = op->getOperands();
    auto attrs = op->getAttrs();
    auto new_op = rewriter.create<tosa::MulOp>(loc, types, operands, attrs);
    new_op->setAttr("shift", rewriter.getI8IntegerAttr(0));
    rewriter.replaceOp(op, new_op);
  };
};

struct LLHDivOpToTosa : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;
  LogicalResult match(DivOp op) const final { return success(); }
  void rewrite(DivOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto types = op->getResultTypes();
    auto rhs = op.getRhs();
    auto lhs = op.getLhs();
    auto new_rhs = rewriter.create<tosa::ReciprocalOp>(loc, rhs.getType(), rhs);
    auto attrs = op->getAttrs();
    auto new_op = rewriter.create<tosa::MulOp>(loc, types,
                                               ValueRange{lhs, new_rhs}, attrs);
    new_op->setAttr("shift", rewriter.getI8IntegerAttr(0));
    rewriter.replaceOp(op, new_op);
  };
};

}  // namespace
//===----------------------------------------------------------------------===//
using LLHAddOpToTosa = SimplyFullLowing<AddOp, tosa::AddOp>;
using LLHSubOpToTosa = SimplyFullLowing<SubOp, tosa::SubOp>;
using LLHConstantOpToTosa = SimplyFullLowing<ConstantOp, tosa::ConstOp>;
// pass defination
//===----------------------------------------------------------------------===//
LLC_DEFINE_CONVERSION_PASS(
    ConvertLLHToTosa,
    {
      LLC_ADD_CONVERSION(LLHAddOpToTosa);
      LLC_ADD_CONVERSION(LLHSubOpToTosa);
      LLC_ADD_CONVERSION(LLHConstantOpToTosa)
      LLC_ADD_CONVERSION(LLHMulOpToTosa);
      LLC_ADD_CONVERSION(LLHDivOpToTosa);
      LLC_ADD_CONVERSION(LLHConvOpToTosa);
    },
    {
      target.addDynamicallyLegalOp<ConstantOp>(check_const_legal);
      target.addIllegalOp<AddOp>();
      target.addIllegalOp<SubOp>();
      target.addIllegalOp<MulOp>();
      target.addIllegalOp<DivOp>();
      target.addIllegalOp<ConvOp>();
      target.addLegalDialect<mlir::tosa::TosaDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
    },
    {
      auto shaped_repalce = [](ShapedType type) { return type; };
      auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
      converter.addConversion(ranked_tensor_replace);
      converter.addConversion(shaped_repalce);
    })
