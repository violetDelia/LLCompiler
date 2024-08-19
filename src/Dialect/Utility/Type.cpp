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
  CHECK(UTILITY, mlir::isa<mlir::ShapedType>(type));
  return mlir::cast<mlir::ShapedType>(type).getShape().vec();
}

std::vector<int64_t> getRankTensorFrom(const mlir::Type& type) {
  CHECK(UTILITY, mlir::isa<mlir::RankedTensorType>(type));
  return mlir::cast<mlir::RankedTensorType>(type).getShape().vec();
}

int64_t getElementSizeFrom(const mlir::ShapedType& shape_type) {
  auto rank = shape_type.getRank();
  if (rank == 0) return 0;
  int64_t element_size = 1;
  for (size_t i = 0; i < rank; ++i) {
    element_size *= shape_type.getDimSize(i);
  }
  return element_size;
}
mlir::ex::Layout getLayoutFrom(const mlir::Value& value) {
  auto tensor = mlir::cast_or_null<mlir::RankedTensorType>(value.getType());
  CHECK(UTILITY, tensor) << "value is not mlir::RankedTensorType";
  auto encode =
      mlir::cast_or_null<mlir::ex::EncodingAttr>(tensor.getEncoding());
  CHECK(UTILITY, encode) << "tensor not have mlir::ex::EncodingAttr";
  return encode.getLayout();
}

}  // namespace llc
