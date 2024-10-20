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
#include <functional>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
namespace mlir::llh {

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

OpFoldResult DimOp::fold(FoldAdaptor adaptor) {
  auto input = getInput();
  if (!isa<RankedTensorType>(input.getType())) return {};
  auto maybe_const_dim = getDim();
  if (!llh::isConstIntegerValue(maybe_const_dim)) return {};
  auto type = llc::getRankTensorFrom(input);
  auto dim = llh::getConstIntegerValue(maybe_const_dim);
  if (type.isDynamicDim(dim)) return {};
  return IntegerAttr::get(IntegerType::get(getContext(), 64),
                          type.getDimSize(dim));
}

namespace {
static bool isSplatZero(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APFloat>().isZero();
  }
  if (llvm::isa<IntegerType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APInt>().isZero();
  }
  return false;
}

static bool isSplatOne(Type elemType, DenseElementsAttr val) {
  if (llvm::isa<FloatType>(elemType)) {
    return val && val.isSplat() &&
           (val.getSplatValue<APFloat>().convertToDouble() == 1);
  }
  if (llvm::isa<IntegerType>(elemType)) {
    return val && val.isSplat() && val.getSplatValue<APInt>().isAllOnes();
  }
  return false;
}

DenseElementsAttr splatDenseBinaryFolder(
    DenseElementsAttr lhs, DenseElementsAttr rhs, RankedTensorType returnTy,
    function_ref<APInt(llvm::APInt, llvm::APInt)> int_calculate,
    function_ref<APFloat(llvm::APFloat, llvm::APFloat)> float_calculate) {
  if (rhs && lhs && rhs.isSplat() && lhs.isSplat()) {
    auto lhs_ele_type = llvm::cast<ShapedType>(lhs.getType()).getElementType();
    auto rhs_ele_type = llvm::cast<ShapedType>(rhs.getType()).getElementType();
    if (lhs_ele_type != rhs_ele_type) return {};
    if (llvm::isa<IntegerType>(lhs_ele_type)) {
      APInt l = lhs.getSplatValue<APInt>();
      APInt r = rhs.getSplatValue<APInt>();
      auto result = int_calculate(l, r);
      return DenseElementsAttr::get(returnTy, result);
    }
    if (llvm::isa<FloatType>(lhs_ele_type)) {
      APFloat l = lhs.getSplatValue<APFloat>();
      APFloat r = rhs.getSplatValue<APFloat>();
      auto result = float_calculate(l, r);
      auto c = DenseElementsAttr::get(returnTy, result);
      return DenseElementsAttr::get(returnTy, result);
    }
  }
  return {};
}
}  // namespace

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto res_type = getType();
  if (!isa<IntegerType, FloatType, RankedTensorType>(res_type)) return {};
  if (isa<IntegerType>(res_type)) {
    // add(x, 0) -> x
    if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();
    // add(0, x) -> x
    if (matchPattern(adaptor.getLhs(), m_Zero())) return getRhs();
    // add(sub(a, b), b) -> a
    if (auto sub = getLhs().getDefiningOp<SubOp>())
      if (getRhs() == sub.getRhs()) return sub.getLhs();
    // add(b, sub(a, b)) -> a
    if (auto sub = getRhs().getDefiningOp<SubOp>())
      if (getLhs() == sub.getRhs()) return sub.getLhs();
    return constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APInt &a, const APInt &b) { return a + b; });
  }
  if (isa<FloatType>(res_type)) {
    // add(x, 0) -> x
    if (matchPattern(adaptor.getRhs(), m_AnyZeroFloat())) return getLhs();
    // add(0, x) -> x
    if (matchPattern(adaptor.getLhs(), m_Zero())) return getRhs();
    // add(sub(a, b), b) -> a
    if (auto sub = getLhs().getDefiningOp<SubOp>())
      if (getRhs() == sub.getRhs()) return sub.getLhs();
    // add(b, sub(a, b)) -> a
    if (auto sub = getRhs().getDefiningOp<SubOp>())
      if (getLhs() == sub.getRhs()) return sub.getLhs();
    return constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APFloat &a, const APFloat &b) { return a + b; });
  }
  if (isa<RankedTensorType>(res_type)) {
    auto lhs_type = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhs_type = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
    auto result_type = llvm::dyn_cast<RankedTensorType>(getType());
    if (!lhs_type.getElementType().isIntOrIndexOrFloat() ||
        !rhs_type.getElementType().isIntOrIndexOrFloat())
      return {};
    auto lhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getLhs());
    auto rhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getRhs());
    // add(x, 0) -> x
    if (lhs_type == result_type &&
        isSplatZero(result_type.getElementType(), rhs_attr))
      return getLhs();
    // add(0, x) -> x
    if (rhs_type == result_type &&
        isSplatZero(result_type.getElementType(), lhs_attr))
      return getRhs();
    if (!lhs_attr || !rhs_attr) return {};
    return splatDenseBinaryFolder(
        lhs_attr, rhs_attr, result_type,
        [](const APInt &a, const APInt &b) { return a + b; },
        [](const APFloat &a, const APFloat &b) { return a + b; });
  }
  return {};
};

// sub(b,sub(a,b)) = b-(b-a) = b-b+a = a
OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto res_type = getType();
  if (!isa<IntegerType, FloatType, RankedTensorType>(res_type)) return {};
  if (isa<IntegerType>(res_type)) {
    // sub(x,x) -> 0
    if (getOperand(0) == getOperand(1))
      return Builder(getContext()).getZeroAttr(res_type);
    // sub(x,0) -> x
    if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();
    if (auto add = getLhs().getDefiningOp<AddOp>()) {
      // sub(add(a, b), b) -> a
      if (getRhs() == add.getRhs()) return add.getLhs();
      // sub(add(a, b), a) -> b
      if (getRhs() == add.getLhs()) return add.getRhs();
    }
    return constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APInt &a, const APInt &b) { return a - b; });
  }
  if (isa<FloatType>(res_type)) {
    // sub(x,x) -> 0
    if (getOperand(0) == getOperand(1))
      return Builder(getContext()).getZeroAttr(res_type);
    // sub(x,0) -> x
    if (matchPattern(adaptor.getRhs(), m_AnyZeroFloat())) return getLhs();
    if (auto add = getLhs().getDefiningOp<AddOp>()) {
      // sub(add(a, b), b) -> a
      if (getRhs() == add.getRhs()) return add.getLhs();
      // sub(add(a, b), a) -> b
      if (getRhs() == add.getLhs()) return add.getRhs();
    }
    return constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APFloat &a, const APFloat &b) { return a - b; });
  }
  if (isa<RankedTensorType>(res_type)) {
    auto lhs_type = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhs_type = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
    auto result_type = llvm::dyn_cast<RankedTensorType>(getType());
    if (!lhs_type.getElementType().isIntOrIndexOrFloat() ||
        !rhs_type.getElementType().isIntOrIndexOrFloat())
      return {};
    auto lhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getLhs());
    auto rhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getRhs());
    // sub(x, 0) -> x
    if (lhs_type == result_type &&
        isSplatZero(result_type.getElementType(), rhs_attr))
      return getLhs();
    if (!lhs_attr || !rhs_attr) return {};
    return splatDenseBinaryFolder(
        lhs_attr, rhs_attr, result_type,
        [](const APInt &a, const APInt &b) { return a - b; },
        [](const APFloat &a, const APFloat &b) { return a - b; });
  }
  return {};
};

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto res_type = getType();
  if (!isa<IntegerType, FloatType, RankedTensorType>(res_type)) return {};
  if (isa<IntegerType>(res_type)) {
    // div (x, 1) -> x
    if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();
    // skip div (x, 0)
    if (matchPattern(adaptor.getRhs(), m_Zero())) return {};
    return constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APInt &a, const APInt &b) { return a.udiv(b); });
  }
  if (isa<FloatType>(res_type)) {
    // div (x, 1) -> x
    if (matchPattern(adaptor.getRhs(), m_OneFloat())) return getLhs();
    // skip div (x, 0)
    if (matchPattern(adaptor.getRhs(), m_AnyZeroFloat())) return {};
    return constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APFloat &a, const APFloat &b) { return a / b; });
  }
  if (isa<RankedTensorType>(res_type)) {
    auto lhs_type = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhs_type = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
    auto result_type = llvm::dyn_cast<RankedTensorType>(getType());
    if (!lhs_type.getElementType().isIntOrIndexOrFloat() ||
        !rhs_type.getElementType().isIntOrIndexOrFloat())
      return {};
    auto lhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getLhs());
    auto rhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getRhs());
    // div(x, 1) -> x
    if (lhs_type == result_type &&
        isSplatOne(result_type.getElementType(), rhs_attr))
      return getLhs();
    // skip 0;
    if (isSplatZero(result_type.getElementType(), rhs_attr) ||
        isSplatZero(result_type.getElementType(), lhs_attr))
      return {};
    if (!lhs_attr || !rhs_attr) return {};
    return splatDenseBinaryFolder(
        lhs_attr, rhs_attr, result_type,
        [](const APInt &a, const APInt &b) { return a.udiv(b); },
        [](const APFloat &a, const APFloat &b) { return a / b; });
  }
  return {};
};

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto res_type = getType();
  if (!isa<IntegerType, FloatType, RankedTensorType>(res_type)) return {};
  if (isa<IntegerType>(res_type)) {
    // mul(x, 0) -> 0
    if (matchPattern(adaptor.getRhs(), m_Zero())) return getRhs();
    // mul(x, 1) -> x
    if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();
    return constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APInt &a, const APInt &b) { return a * b; });
  }
  if (isa<FloatType>(res_type)) {
    // mul(x, 0) -> 0
    if (matchPattern(adaptor.getRhs(), m_AnyZeroFloat())) return getRhs();
    // mul(x, 1) -> x
    if (matchPattern(adaptor.getRhs(), m_OneFloat())) return getLhs();
    return constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(
        adaptor.getOperands(),
        [](const APFloat &a, const APFloat &b) { return a * b; });
  }
  if (isa<RankedTensorType>(res_type)) {
    auto lhs_type = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhs_type = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
    auto result_type = llvm::dyn_cast<RankedTensorType>(getType());
    if (!lhs_type.getElementType().isIntOrIndexOrFloat() ||
        !rhs_type.getElementType().isIntOrIndexOrFloat())
      return {};
    auto lhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getLhs());
    auto rhs_attr =
        llvm::dyn_cast_if_present<DenseElementsAttr>(adaptor.getRhs());
    // mul(x, 0) -> 0
    if (lhs_type == result_type &&
        isSplatZero(result_type.getElementType(), rhs_attr))
      return getRhs();
    // mul(x, 1) -> x
    if (lhs_type == result_type &&
        isSplatOne(result_type.getElementType(), rhs_attr))
      return getLhs();
    if (!lhs_attr || !rhs_attr) return {};
    return splatDenseBinaryFolder(
        lhs_attr, rhs_attr, result_type,
        [](const APInt &a, const APInt &b) { return a * b; },
        [](const APFloat &a, const APFloat &b) { return a / b; });
  }
  return {};
};

}  // namespace mlir::llh
