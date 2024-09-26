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
#include "llcompiler/Dialect/Utility/Type.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llcompiler/Dialect/IRExtension/IR/Attrs.h"
#include "llcompiler/Dialect/IRExtension/IR/Enums.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace llc {

std::vector<int64_t> getShapeFrom(const mlir::Type& type) {
  CHECK(::llc::UTILITY, mlir::isa<mlir::ShapedType>(type));
  return mlir::cast<mlir::ShapedType>(type).getShape().vec();
}

std::vector<int64_t> getRankTensorFrom(const mlir::Type& type) {
  CHECK(::llc::UTILITY, mlir::isa<mlir::RankedTensorType>(type));
  return mlir::cast<mlir::RankedTensorType>(type).getShape().vec();
}

int64_t getElementSizeFrom(const mlir::ShapedType& shapeType) {
  auto rank = shapeType.getRank();
  if (rank == 0) return 0;
  int64_t element_size = 1;
  for (size_t i = 0; i < rank; ++i) {
    element_size *= shapeType.getDimSize(i);
  }
  return element_size;
}
// mlir::ex::Layout getLayoutFrom(const mlir::RankedTensorType& tensor) {
//   auto encode =
//       mlir::cast_or_null<mlir::ex::EncodingAttr>(tensor.getEncoding());
//   CHECK(UTILITY, encode) << "tensor not have mlir::ex::EncodingAttr";
//   return encode.getLayout();
// }

// mlir::RankedTensorType cloneTensorWithEncoding(
//     const mlir::RankedTensorType& tensor, mlir::ex::Layout layout) {
//   auto type = tensor.getElementType();
//   auto shape = tensor.getShape();
//   auto encode = mlir::ex::EncodingAttr::get(tensor.getContext(), layout);
//   return mlir::RankedTensorType::get(shape, type, encode);
// }

std::vector<int64_t> getUnsqueezeShape(const mlir::ShapedType& shapeType,
                                       int dim) {
  auto shape = shapeType.getShape().vec();
  shape.insert(shape.begin() + dim, 1);
  return shape;
}

std::vector<int64_t> getSqueezeShape(const mlir::ShapedType& shapeType,
                                     int dim) {
  auto shape = shapeType.getShape().vec();
  CHECK_GT(UTILITY, shape.size(), 0);
  shape.erase(shape.begin() + dim);
  return shape;
}
mlir::RankedTensorType getUnsqueezeTensor(const mlir::RankedTensorType& tensor,
                                          int dim) {
  // auto encode =
  //     mlir::ex::EncodingAttr::get(tensor.getContext(),
  //     getLayoutFrom(tensor));
  return mlir::RankedTensorType::get(getUnsqueezeShape(tensor),
                                     tensor.getElementType());
}
mlir::RankedTensorType getSqueezeTensor(const mlir::RankedTensorType& tensor,
                                        int dim) {
  // auto encode =
  //     mlir::ex::EncodingAttr::get(tensor.getContext(),
  //     getLayoutFrom(tensor));
  return mlir::RankedTensorType::get(getSqueezeShape(tensor),
                                     tensor.getElementType());
}

}  // namespace llc
