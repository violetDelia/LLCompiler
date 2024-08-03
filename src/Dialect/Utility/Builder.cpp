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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

namespace llc {

#define BUILD_CONST_OP(judge, Ty, Op)                                      \
  if (judge) {                                                             \
    llvm::SmallVector<Ty> vec(value);                                      \
    llvm::ArrayRef<Ty> new_value(vec);                                     \
    auto value_attr = mlir::DenseElementsAttr::get(shape_type, new_value); \
    auto const_op = build_op<Op>(builder, shape_type, value_attr);         \
    return const_op;                                                       \
  }

mlir::tosa::ConstOp create_tosa_const(mlir::OpBuilder *builder,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<double> value,
                                      mlir::Type type) {
  CHECK(UTILITY, type.isIntOrFloat()) << "Invalid element type";
  auto shape_type = mlir::RankedTensorType::get(shape, type);
  BUILD_CONST_OP(type.isInteger(1), bool, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignedInteger(8), int8_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignedInteger(16), int16_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignedInteger(32), int32_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignedInteger(64), int64_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignlessInteger(8), uint8_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignlessInteger(16), uint16_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignlessInteger(32), uint32_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isSignlessInteger(64), uint64_t, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isF32(), float, mlir::tosa::ConstOp)
  BUILD_CONST_OP(type.isF64(), double, mlir::tosa::ConstOp)
  UNIMPLEMENTED(UTILITY);
}

#undef BUILD_CONST_OP

mlir::tensor::ExpandShapeOp expand_to(
    mlir::OpBuilder *builder, mlir::Operation *from, mlir::ShapedType expand_to,
    mlir::ArrayRef<mlir::ReassociationIndices> reassociation) {
  auto expand_op = builder->create<mlir::tensor::ExpandShapeOp>(
      from->getLoc(), expand_to, from->getResult(0), reassociation);
  return expand_op;
}

// return: [tosa::const(value), shape.expand_shape([1]->[target_shape])]
llvm::SmallVector<mlir::Operation *> expand_const_to(
    mlir::OpBuilder *builder, double value, mlir::Type element_type,
    mlir::RankedTensorType target_shape) {
  llvm::SmallVector<mlir::Operation *> outs;
  auto const_op = create_tosa_const(builder, {1}, {value}, element_type);
  mlir::ReassociationIndices reassociations;
  auto rank = target_shape.getRank();
  for (int i = 0; i < rank; ++i) {
    reassociations.push_back(i);
  }
  auto expand_op = expand_to(builder, const_op, target_shape, {reassociations});
  outs.push_back(const_op);
  outs.push_back(expand_op);
  return outs;
}
}  // namespace llc
