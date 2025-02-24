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
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::llh {
#define COMPUTABLE_BINARY_VERIFY(OP)                                          \
  ::llvm::LogicalResult OP::verify() {                                        \
    auto op = getOperation();                                                 \
    auto input1_type = op->getOperand(0).getType();                           \
    auto input2_type = op->getOperand(1).getType();                           \
    auto res_type = op->getResult(0).getType();                               \
    auto input1_is_tensor =                                                   \
        isa<RankedTensorType, UnrankedTensorType>(input1_type);               \
    auto input2_is_tensor =                                                   \
        isa<RankedTensorType, UnrankedTensorType>(input2_type);               \
    auto res_is_tensor = isa<RankedTensorType, UnrankedTensorType>(res_type); \
    if (input1_is_tensor && input2_is_tensor && res_is_tensor) {              \
      return llvm::success();                                                 \
    }                                                                         \
    auto input1_is_int = isa<IntegerType>(input1_type);                       \
    auto input2_is_int = isa<IntegerType>(input2_type);                       \
    auto res_is_int = isa<IntegerType>(res_type);                             \
    if (input1_is_int && input2_is_int && res_is_int) {                       \
      return llvm::success();                                                 \
    }                                                                         \
    auto input1_is_float = isa<FloatType>(input1_type);                       \
    auto input2_is_float = isa<FloatType>(input2_type);                       \
    auto res_is_float = isa<FloatType>(res_type);                             \
    if (input1_is_float && input2_is_float && res_is_float) {                 \
      return llvm::success();                                                 \
    }                                                                         \
    return llvm::failure();                                                   \
  }

COMPUTABLE_BINARY_VERIFY(SubOp)
COMPUTABLE_BINARY_VERIFY(AddOp)
COMPUTABLE_BINARY_VERIFY(DivOp)
COMPUTABLE_BINARY_VERIFY(MulOp)
COMPUTABLE_BINARY_VERIFY(MaxOp)
COMPUTABLE_BINARY_VERIFY(MinOp)
COMPUTABLE_BINARY_VERIFY(PowOp)

::llvm::LogicalResult ScalarCastOP::verify() {
  auto input = getInput();
  RankedTensorType tensor_type;
  Type element_type;
  auto input_type = input.getType();
  auto res_type = getType();
  if (isa<RankedTensorType>(input_type)) {
    tensor_type = cast<RankedTensorType>(input_type);
    element_type = res_type;
  } else if (isa<RankedTensorType>(res_type)) {
    tensor_type = cast<RankedTensorType>(res_type);
    element_type = input_type;
  } else {
    return llvm::failure();
  }
  return llvm::success();
  if (tensor_type.getElementType() == element_type) {
    return llvm::success();
  }
  return llvm::failure();
}
#undef COMPUTABLE_BINARY_VERIFY
}  // namespace mlir::llh
