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
#include "mlir/IR/Operation.h"
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

// Utilitys
namespace mlir::llh {
llh::DimOp buildTensorDim(mlir::Value operand, LLHPatternRewriter* rewrite,
                          size_t dim);
llvm::SmallVector<Value> buildTensorDims(mlir::Value operand,
                                         LLHPatternRewriter* rewrite);

bool shapeIsSame(Value lhs, Value rhs);

LogicalResult checkBinaryNeedReshape(Operation* op);

ReshapeOp ReshapeValueTo(Value lower_value, Value higher_value,
                         LLHPatternRewriter* rewriter);

template <class BinaryOp>
LogicalResult insertReshapeBeforeBinary(Operation* op,
                                        LLHPatternRewriter& rewriter) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  auto res = op->getResult(0);
  auto lhs_tensor = llc::getRankTensorFrom(lhs);
  auto rhs_tensor = llc::getRankTensorFrom(rhs);
  auto lhs_rank = lhs_tensor.getRank();
  auto rhs_rank = rhs_tensor.getRank();
  Value higher_value, lower_value;
  Operation* reshape_op;
  if (lhs_rank > rhs_rank) {
    reshape_op = ReshapeValueTo(rhs, lhs, &rewriter);
    rewriter.replaceOpWithNewOp<BinaryOp>(
        op, TypeRange{res.getType()}, ValueRange{lhs, reshape_op->getResult(0)},
        op->getAttrDictionary().getValue());
  } else {
    reshape_op = ReshapeValueTo(lhs, rhs, &rewriter);
    rewriter.replaceOpWithNewOp<BinaryOp>(
        op, TypeRange{res.getType()}, ValueRange{reshape_op->getResult(0), rhs},
        op->getAttrDictionary().getValue());
  }
  return llvm::success();
}

LogicalResult checkBinaryNeedBroadcast(Operation* op);

BroadCastToOp broadcastValueTo(Value value, Value target_type,
                               LLHPatternRewriter* rewriter);

template <class BinaryOp>
LogicalResult insertBroadcastBeforeBinary(Operation* op,
                                          LLHPatternRewriter& rewriter) {
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  auto result = op->getResult(0);
  auto result_type = llc::getRankTensorFrom(result);
  Value will_be_broadcast;
  Value target_operand;
  Operation* cast_op;
  if (shapeIsSame(lhs, result)) {
    cast_op = broadcastValueTo(rhs, lhs, &rewriter);
    rewriter.replaceOpWithNewOp<BinaryOp>(
        op, TypeRange{result_type}, ValueRange{lhs, cast_op->getResult(0)},
        op->getAttrDictionary().getValue());

  } else if (shapeIsSame(rhs, result)) {
    cast_op = broadcastValueTo(lhs, rhs, &rewriter);
    rewriter.replaceOpWithNewOp<BinaryOp>(
        op, TypeRange{result_type}, ValueRange{cast_op->getResult(0), rhs},
        op->getAttrDictionary().getValue());
  } else {
    FATAL(llc::MLIR_PASS) << "Unexpected result";
    return llvm::failure();
  }
  return llvm::success();
}
}  // namespace mlir::llh
// patterns
namespace mlir::llh {
void populateSymbolCanonicalizePatterns(RewritePatternSet& patterns);
void populateSinkSymbolBindPatterns(RewritePatternSet& patterns);
}  // namespace mlir::llh

// reshape and broadcast
namespace mlir::llh {}  // namespace mlir::llh
#endif                  // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_
