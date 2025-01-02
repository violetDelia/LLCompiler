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
#include <cstddef>
#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
namespace mlir::llh {
llh::DimOp buildTensorDim(mlir::Value operand, LLHPatternRewriter* rewrite,
                          size_t dim) {
  auto loc = operand.getLoc();
  auto dim_const = rewrite->create<ConstantOp>(
      loc, IntegerAttr::get(rewrite->getI64Type(), dim));
  return rewrite->create<DimOp>(loc, ValueRange{operand, dim_const});
}

llvm::SmallVector<Value> buildTensorDims(mlir::Value operand,
                                         LLHPatternRewriter* rewrite) {
  auto tensor = llvm::dyn_cast_or_null<ShapedType>(operand.getType());
  CHECK(llc::MLIR_PASS, tensor);
  auto rank = tensor.getRank();
  auto ranks = SmallVector<Value>();
  for (int i{}; i < rank; i++) {
    ranks.push_back(buildTensorDim(operand, rewrite, i));
  }
  return ranks;
}

bool shapeIsSame(Value lhs, Value rhs) {
  auto lhs_type = llc::getShapeTypeFrom(lhs);
  auto rhs_type = llc::getShapeTypeFrom(rhs);
  if (rhs_type.getRank() != lhs_type.getRank()) return false;
  if (llc::hasEncoding(lhs_type) && llc::hasEncoding(rhs_type)) {
    auto lhs_encoding = llc::getEncodingFrom(lhs);
    auto rhs_encoding = llc::getEncodingFrom(rhs);
    auto lhs_symbols = lhs_encoding.getShapeSymbols();
    auto rhs_symbols = rhs_encoding.getShapeSymbols();
    for (auto [lhs_symbol, rhs_symbol] : llvm::zip(lhs_symbols, rhs_symbols)) {
      if (lhs_symbol != rhs_symbol) return false;
    }
    return true;
  }
  auto lhs_shapes = lhs_type.getShape();
  auto rhs_shapes = rhs_type.getShape();
  for (auto [lhs_shape, rhs_shape] : llvm::zip(lhs_shapes, rhs_shapes)) {
    if (lhs_shape != rhs_shape) return false;
  }
  return true;
}
ReshapeOp ReshapeValueTo(Value lower_value, Value higher_value,
                         LLHPatternRewriter* rewriter) {
  auto loc = lower_value.getLoc();
  auto higher_shapes = llc::getShapeFrom(higher_value);
  auto lower_shapes = llc::getShapeFrom(lower_value);
  auto higher_rank = higher_shapes.size();
  auto lower_tensor = llc::getRankTensorFrom(lower_value);
  auto one_const = rewriter->create<ConstantOp>(
      loc, IntegerAttr::get(rewriter->getI64Type(), 1));
  auto reshape_dims = llvm::SmallVector<mlir::Value>();
  auto reshape_shapes = llvm::SmallVector<int64_t>();
  reshape_shapes.assign(higher_rank, 1);
  reshape_dims.assign(higher_rank, one_const);
  for (int64_t i = higher_shapes.size() - 1, j = lower_shapes.size() - 1;
       i >= 0 && j >= 0; i--, j--) {
    auto higher_dim = higher_shapes[i];
    auto lower_dim = lower_shapes[j];
    if (lower_dim == 1 && (higher_dim > 1 || higher_dim < 0)) {
    } else if (((lower_dim > 1 || lower_dim < 0) && higher_dim == 1) ||
               (lower_dim == higher_dim)) {
      reshape_shapes[i] = lower_dim;
      reshape_dims[i] = buildTensorDim(lower_value, rewriter, j);
    } else {
      WRONG(llc::MLIR) << "Invalid broadcast case";
      return nullptr;
    }
  }
  auto reshape_res =
      RankedTensorType::get(reshape_shapes, lower_tensor.getElementType());
  auto reshape = rewriter->create<llh::ReshapeOp>(loc, reshape_res, lower_value,
                                                  reshape_dims);
  return reshape;
};

LogicalResult checkBinaryNeedReshape(Operation* op) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (!isa<RankedTensorType>(lhs.getType())) return llvm::failure();
  if (!isa<RankedTensorType>(rhs.getType())) return llvm::failure();
  auto lhs_rank = llc::getRankTensorFrom(lhs).getRank();
  auto rhs_rank = llc::getRankTensorFrom(rhs).getRank();
  if (lhs_rank == rhs_rank) return llvm::failure();
  return llvm::success();
}

BroadCastToOp broadcastValueTo(Value value, Value target,
                               LLHPatternRewriter* rewriter) {
  auto context = rewriter->getContext();
  auto loc = value.getLoc();
  auto before_braodcast_type = llc::getRankTensorFrom(value);
  auto target_type = llc::getRankTensorFrom(target);
  llvm::SmallVector<int64_t> cast_dims;
  llvm::SmallVector<int64_t> noexpand_dims;
  llvm::SmallVector<int64_t> expand_dims;
  auto before_type = llc::getRankTensorFrom(before_braodcast_type);
  for (size_t i = 0; i < target_type.getRank(); i++) {
    cast_dims.push_back(i);
    if (before_type.isDynamicDim(i)) continue;
    if (before_type.getDimSize(i) == target_type.getDimSize(i))
      noexpand_dims.push_back(i);
    else if (before_type.getDimSize(i) == 1)
      expand_dims.push_back(i);
  }
  auto res_type = RankedTensorType::get(target_type.getShape(),
                                        before_braodcast_type.getElementType());
  auto cast_op = rewriter->create<BroadCastToOp>(
      loc, res_type, value, buildTensorDims(target, rewriter), cast_dims,
      DenseI64ArrayAttr::get(context, expand_dims),
      DenseI64ArrayAttr::get(context, noexpand_dims));
  return cast_op;
}

LogicalResult checkBinaryNeedBroadcast(Operation* op) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (!isa<RankedTensorType>(lhs.getType())) return llvm::failure();
  auto lhs_type = llc::getRankTensorFrom(lhs);
  auto rhs_type = llc::getRankTensorFrom(rhs);
  auto lhs_rank = lhs_type.getRank();
  auto rhs_rank = rhs_type.getRank();
  if (lhs_rank != rhs_rank) return llvm::failure();
  if (llc::equalShape(lhs_type, rhs_type)) return llvm::failure();
  return llvm::success();
}
}  // namespace mlir::llh