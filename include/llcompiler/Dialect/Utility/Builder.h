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
#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_BUILDER_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_BUILDER_H_
#include <string>
#include <utility>

#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace llc {

template <class Op, class... Args>
Op build_op(mlir::OpBuilder *builder, Args... args) {
  return builder->create<Op>(builder->getUnknownLoc(),
                             std::forward<Args>(args)...);
}

template <class Vlaue_Type>
mlir::Operation *create_broadcast_const_to(mlir::OpBuilder *builder,
                                           const mlir::Type &type,
                                           mlir::ArrayRef<Vlaue_Type> shape,
                                           mlir::ArrayRef<Vlaue_Type> value) {
  if (!mlir::isa<mlir::ShapedType>(type)) return nullptr;
  auto target_shape = mlir::cast<mlir::ShapedType>(type);
  target_shape.dump();
  auto shape_type =
      mlir::RankedTensorType::get(shape, target_shape.getElementType());
  shape_type.dump();
  auto value_attr = mlir::DenseElementsAttr::get(shape_type, value);
  value_attr.dump();
  auto const_op =
      build_op<mlir::tosa::ConstOp>(builder, shape_type, value_attr);
  return const_op;
}
}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_BUILDER_H_
