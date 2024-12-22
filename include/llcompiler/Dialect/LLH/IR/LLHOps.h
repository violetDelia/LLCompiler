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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHTypesImpl.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Interfaces/BraodcastableOpInterfaces.h"
#include "llcompiler/Interfaces/SymbolShapeOpInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define PLACEHOLD_FOR_FIX_HEADER
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h.inc"
#define GET_OP_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHOps.h.inc"
#undef PLACEHOLD_FOR_FIX_HEADER

namespace mlir::llh {
void populateSymbolCanonicalizePatterns(RewritePatternSet& patterns);
void populateSinkSymbolBindPatterns(RewritePatternSet& patterns);

namespace detail {
llh::DimOp buildTensorDim(mlir::Value operand, LLHPatternRewriter* rewrite,
                          size_t dim);
llvm::SmallVector<Value> buildTensorDims(mlir::Value operand,
                                         LLHPatternRewriter* rewrite);
}  // namespace detail

LogicalResult checkBinaryNeedReshape(Operation* op);

template <class BinaryOp>
LogicalResult insertReshapeBeforeBinary(Operation* op,
                                        LLHPatternRewriter& rewriter) {
  auto loc = op->getLoc();
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
      reshape_dims[i] = detail::buildTensorDim(lower_value, &rewriter, j);
    } else {
      WRONG(llc::MLIR) << "Invalid broadcast case";
      return llvm::failure();
    }
  }
  auto reshape_res =
      RankedTensorType::get(reshape_shapes, lhs_tensor.getElementType());
  auto reshape = rewriter.create<llh::ReshapeOp>(loc, reshape_res, lower_value,
                                                 reshape_dims);
  if (lhs_rank > rhs_rank) {
    auto new_op = rewriter.replaceOpWithNewOp<BinaryOp>(
        op, TypeRange{res.getType()}, ValueRange{lhs, reshape},
        op->getAttrDictionary().getValue());
  } else {
    auto new_op = rewriter.replaceOpWithNewOp<BinaryOp>(
        op, TypeRange{res.getType()}, ValueRange{reshape, rhs},
        op->getAttrDictionary().getValue());
  }
  return llvm::success();
}

LogicalResult checkBinaryNeedBroadcast(Operation* op);

template <class BinaryOp>
LogicalResult insertBroadcastBeforeBinary(Operation* op,
                                          LLHPatternRewriter& rewriter) {
  auto context = rewriter.getContext();
  auto loc = op->getLoc();
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
    return llvm::failure();
  }
  auto before_braodcast_type = llc::getRankTensorFrom(will_be_broadcast);
  llvm::SmallVector<int64_t> cast_dims;
  llvm::SmallVector<int64_t> noexpand_dims;
  llvm::SmallVector<int64_t> expand_dims;
  auto before_type = llc::getRankTensorFrom(before_braodcast_type);
  for (size_t i = 0; i < result_type.getRank(); i++) {
    cast_dims.push_back(i);
    if (before_type.isDynamicDim(i)) continue;
    if (before_type.getDimSize(i) == result_type.getDimSize(i))
      noexpand_dims.push_back(i);
    else if (before_type.getDimSize(i) == 1)
      expand_dims.push_back(i);
  }
  auto cast_op = rewriter.create<BroadCastToOp>(
      loc, result_type, will_be_broadcast,
      detail::buildTensorDims(target_operand, &rewriter), cast_dims,
      DenseI64ArrayAttr::get(context, expand_dims),
      DenseI64ArrayAttr::get(context, noexpand_dims));
  if (lhs_type == result_type) {
    rewriter.replaceOpWithNewOp<BinaryOp>(op, result_type,
                                          ValueRange{lhs, cast_op});
  } else {
    rewriter.replaceOpWithNewOp<BinaryOp>(op, result_type,
                                          ValueRange{cast_op, rhs});
  }
  return llvm::success();
}
}  // namespace mlir::llh
#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_
