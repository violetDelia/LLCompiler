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

#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

namespace llc {

#define BUILD_CONST_OP(judge, Ty, Op, loc)                                 \
  if (judge) {                                                             \
    llvm::SmallVector<Ty> vec(value);                                      \
    llvm::ArrayRef<Ty> new_value(vec);                                     \
    auto value_attr = mlir::DenseElementsAttr::get(shape_type, new_value); \
    auto const_op = builder->create<Op>(loc, shape_type, value_attr);      \
    return const_op;                                                       \
  }

mlir::tosa::ConstOp create_tosa_const(mlir::OpBuilder *builder,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<double> value,
                                      mlir::Type type,
                                      const mlir::Location &loc) {
  CHECK(UTILITY, type.isIntOrFloat()) << "Invalid element type";
  auto shape_type = mlir::RankedTensorType::get(shape, type);
  BUILD_CONST_OP(type.isInteger(1), bool, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignedInteger(8), int8_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignedInteger(16), int16_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignedInteger(32), int32_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignedInteger(64), int64_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignlessInteger(8), uint8_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignlessInteger(16), uint16_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignlessInteger(32), uint32_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isSignlessInteger(64), uint64_t, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isF32(), float, mlir::tosa::ConstOp, loc)
  BUILD_CONST_OP(type.isF64(), double, mlir::tosa::ConstOp, loc)
  UNIMPLEMENTED(UTILITY);
}

#undef BUILD_CONST_OP

mlir::tensor::ExpandShapeOp expand_to(
    mlir::OpBuilder *builder, mlir::Operation *from, mlir::ShapedType expand_to,
    mlir::ArrayRef<mlir::ReassociationIndices> reassociation,
    const mlir::Location &loc) {
  auto expand_op = builder->create<mlir::tensor::ExpandShapeOp>(
      loc, expand_to, from->getResult(0), reassociation);
  return expand_op;
}

}  // namespace llc
