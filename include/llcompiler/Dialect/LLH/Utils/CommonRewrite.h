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
//

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_COMMONREWRITE_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_COMMONREWRITE_H_
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/Operation.h"
namespace mlir::llh {
template <class BinaryOp>
struct SimplyBinaryOpInsertBraodcast : public LLHOpRewritePattern<BinaryOp> {
  using LLHOpRewritePattern<BinaryOp>::LLHOpRewritePattern;
  LogicalResult match(BinaryOp op) const final {
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
  void rewrite(BinaryOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhs_type = llc::getRankTensorFrom(lhs);
    auto rhs_type = llc::getRankTensorFrom(rhs);
    auto result = op->getResult(0);
    auto result_type = llc::getRankTensorFrom(result);
    Value will_be_broadcast;
    Value target_operand;
    if (lhs_type == result_type) {
      will_be_broadcast = rhs;
      target_operand = lhs;

    } else if (rhs_type == result_type) {
      will_be_broadcast = lhs;
      target_operand = rhs;
    } else {
      FATAL(llc::MLIR_PASS) << "Unexpected result";
    }
    auto before_braodcast_type = llc::getRankTensorFrom(will_be_broadcast);
    llvm::SmallVector<int64_t> cast_dims;
    auto before_type = llc::getRankTensorFrom(before_braodcast_type);
    for (size_t i = 0; i < result_type.getRank(); i++) {
      if (before_type.getDimSize(i) != result_type.getDimSize(i)) {
        cast_dims.push_back(i);
      }
    }
    auto cast_op = rewriter.create<BroadCastToOp>(
        loc, result_type, will_be_broadcast,
        llh::buildTensorDims(target_operand, &rewriter), cast_dims,
        DenseI64ArrayAttr{}, DenseI64ArrayAttr{});
    if (lhs_type == result_type) {
      rewriter.replaceOpWithNewOp<BinaryOp>(op, result_type,
                                            ValueRange{lhs, cast_op});
    } else {
      rewriter.replaceOpWithNewOp<BinaryOp>(op, result_type,
                                            ValueRange{cast_op, rhs});
    }
  }
};

template <class BinaryOp>
struct SimplyBinaryOpReshape : public LLHOpRewritePattern<BinaryOp> {
  using LLHOpRewritePattern<BinaryOp>::LLHOpRewritePattern;
  LogicalResult match(BinaryOp op) const final {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    if (!isa<RankedTensorType>(lhs.getType())) return llvm::failure();
    if (!isa<RankedTensorType>(rhs.getType())) return llvm::failure();
    auto lhs_rank = llc::getRankTensorFrom(lhs).getRank();
    auto rhs_rank = llc::getRankTensorFrom(rhs).getRank();
    if (lhs_rank == rhs_rank) return llvm::failure();
    return llvm::success();
  }
  void rewrite(BinaryOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto res = op->getResult(0);
    auto lhs_tensor = llc::getRankTensorFrom(lhs);
    auto rhs_tensor = llc::getRankTensorFrom(rhs);
    auto lhs_rank = lhs_tensor.getRank();
    auto rhs_rank = rhs_tensor.getRank();
    Value higher_value, lower_value;
    if (lhs_rank > rhs_rank) {
      higher_value = lhs;
      lower_value = rhs;
    } else {
      higher_value = rhs;
      lower_value = lhs;
    }
    auto higher_shapes = llc::getShapeFrom(higher_value);
    auto lower_shapes = llc::getShapeFrom(lower_value);
    auto higher_rank = higher_shapes.size();
    auto lower_rank = lower_shapes.size();
    auto one_const = rewriter.create<ConstantOp>(
        loc, IntegerAttr::get(rewriter.getI64Type(), 1));
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
        reshape_dims[i] = llh::buildTensorDim(lower_value, &rewriter, j);
      } else {
        WRONG(llc::MLIR) << "Invalid broadcast case";
        return;
      }
    }
    auto reshape_res =
        RankedTensorType::get(reshape_shapes, lhs_tensor.getElementType());
    auto reshape = rewriter.create<llh::ReshapeOp>(loc, reshape_res,
                                                   lower_value, reshape_dims);
    if (lhs_rank > rhs_rank) {
      rewriter.replaceOpWithNewOp<BinaryOp>(op, TypeRange{res.getType()},
                                            ValueRange{lhs, reshape},
                                            op->getAttrDictionary().getValue());
    } else {
      rewriter.replaceOpWithNewOp<BinaryOp>(op, TypeRange{res.getType()},
                                            ValueRange{reshape, rhs},
                                            op->getAttrDictionary().getValue());
    }
  }
};

}  // namespace mlir::llh
#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_COMMONREWRITE_H_
