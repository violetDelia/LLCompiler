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
#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_TYPE_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_TYPE_H_
#include <cstdint>
#include <vector>

#include "llcompiler/Dialect/IRExtension/IR/Enums.h"
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace llc {
std::vector<int64_t> getShapeFrom(mlir::Type shapeType);
std::vector<int64_t> getShapeFrom(mlir::Value value);
mlir::RankedTensorType getRankTensorFrom(mlir::Type type);
mlir::RankedTensorType getRankTensorFrom(mlir::Value value);
bool hasEncoding(mlir::Type type);
bool hasEncoding(mlir::Value value);
::mlir::llh::EncodingAttr getEncodingFrom(mlir::Type type);
::mlir::llh::EncodingAttr getEncodingFrom(mlir::Value value);
int64_t getElementSizeFrom(mlir::ShapedType shapeType);
mlir::DenseElementsAttr genZoreElementAttr(mlir::Value value);
bool equalShape(mlir::ShapedType lhs,mlir::ShapedType rhs);
// mlir::ex::Layout getLayoutFrom(mlir::RankedTensorType value);
// mlir::RankedTensorType cloneTensorWithEncoding(
//     mlir::RankedTensorType value, mlir::ex::Layout layout);
std::vector<int64_t> getUnsqueezeShape(mlir::ShapedType shapeType, int dim = 0);
std::vector<int64_t> getSqueezeShape(mlir::ShapedType shapeType, int dim = 0);
mlir::RankedTensorType getUnsqueezeTensor(mlir::RankedTensorType tensor,
                                          int dim = 0);
mlir::RankedTensorType getSqueezeTensor(mlir::RankedTensorType tensor,
                                        int dim = 0);
}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_TYPE_H_
