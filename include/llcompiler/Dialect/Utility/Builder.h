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
#include <string>
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

namespace llc {
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
