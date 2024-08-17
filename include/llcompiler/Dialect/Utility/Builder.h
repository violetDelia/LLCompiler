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
#include <cstdint>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include <string>

namespace llc {
#define ADD_ATTR_FUNC(name, input_type, attr_type)                       \
  void add_##name##_attr(mlir::Operation *op, llvm::StringRef attr_name, \
                         input_type value) {                             \
    op->setAttr(attr_name, attr_type::get(op->getContext(), value));     \
  }

ADD_ATTR_FUNC(array_i64, llvm::SmallVector<int64_t>, mlir::DenseI64ArrayAttr)
ADD_ATTR_FUNC(array_i64, llvm::ArrayRef<int64_t>, mlir::DenseI64ArrayAttr)
ADD_ATTR_FUNC(array_i1, llvm::SmallVector<bool>, mlir::DenseBoolArrayAttr)
ADD_ATTR_FUNC(array_i1, llvm::ArrayRef<bool>, mlir::DenseBoolArrayAttr)
ADD_ATTR_FUNC(bool, bool, mlir::BoolAttr)
ADD_ATTR_FUNC(string, llvm::StringRef, mlir::StringAttr)
ADD_ATTR_FUNC(string, const char *, mlir::StringAttr)
#undef ADD_ATTR_FUNC

template <class Op, class... Args>
Op build_op(mlir::OpBuilder *builder, Args... args) {
  return builder->create<Op>(builder->getUnknownLoc(),
                             std::forward<Args>(args)...);
}
mlir::tosa::ConstOp create_tosa_const(mlir::OpBuilder *builder,
                                      llvm::ArrayRef<int64_t> shape,
                                      llvm::ArrayRef<double> value,
                                      mlir::Type element_type,
                                      const mlir::Location &loc);

}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_BUILDER_H_
