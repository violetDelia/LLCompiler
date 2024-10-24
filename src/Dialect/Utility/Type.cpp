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
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace llc {

std::vector<int64_t> getShapeFrom(mlir::Type type) {
  CHECK(::llc::UTILITY, mlir::isa<mlir::ShapedType>(type));
  return mlir::cast<mlir::ShapedType>(type).getShape().vec();
}

std::vector<int64_t> getShapeFrom(mlir::Value value) {
  auto type = value.getType();
  return getShapeFrom(type);
}

mlir::RankedTensorType getRankTensorFrom(mlir::Type type) {
  CHECK(::llc::UTILITY, mlir::isa<mlir::RankedTensorType>(type));
  return mlir::cast<mlir::RankedTensorType>(type);
}

mlir::RankedTensorType getRankTensorFrom(mlir::Value value) {
  auto type = value.getType();
  return getRankTensorFrom(type);
}

bool hasEncoding(mlir::Type type) {
  auto tensor = mlir::dyn_cast_or_null<mlir::RankedTensorType>(type);
  if (!tensor) return false;
  auto has_encoding =
      mlir::dyn_cast_or_null<mlir::llh::EncodingAttr>(tensor.getEncoding());
  if (!has_encoding) return false;
  return true;
}

bool hasEncoding(mlir::Value value) {
  auto type = value.getType();
  return hasEncoding(type);
}

::mlir::llh::EncodingAttr getEncodingFrom(mlir::Type type) {
  CHECK(llc::UTILITY, llvm::isa<mlir::RankedTensorType>(type));
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
  auto encoding = tensor.getEncoding();
  CHECK(llc::UTILITY, llvm::isa<mlir::llh::EncodingAttr>(encoding));
  return llvm::dyn_cast<mlir::llh::EncodingAttr>(encoding);
}
::mlir::llh::EncodingAttr getEncodingFrom(mlir::Value value) {
  auto type = value.getType();
  return getEncodingFrom(type);
}

int64_t getElementSizeFrom(mlir::ShapedType shapeType) {
  auto rank = shapeType.getRank();
  if (rank == 0) return 0;
  int64_t element_size = 1;
  for (size_t i = 0; i < rank; ++i) {
    element_size *= shapeType.getDimSize(i);
  }
  return element_size;
}

bool equalShape(mlir::ShapedType lhs, mlir::ShapedType rhs) {
  auto lhs_rank = lhs.getRank();
  auto rhs_rank = rhs.getRank();
  if (lhs_rank != rhs_rank) return false;
  for (int i = 0; i < lhs_rank; i++) {
    if (lhs.getDimSize(i) != rhs.getDimSize(i)) return false;
  }
  return true;
};

#define BUILD_ATTR(judge, Ty, shape)                        \
  if (judge) {                                              \
    llvm::ArrayRef<Ty> value(0);                            \
    auto attr = mlir::DenseElementsAttr::get(shape, value); \
    return attr;                                            \
  }

mlir::DenseElementsAttr genZoreElementAttr(mlir::Value value) {
  CHECK(llc::MLIR, llvm::isa<mlir::RankedTensorType>(value.getType()));
  auto tensor = llvm::cast<mlir::RankedTensorType>(value.getType());
  auto type = tensor.getElementType();
  BUILD_ATTR(type.isInteger(1), bool, tensor)
  BUILD_ATTR(type.isSignedInteger(8), int8_t, tensor)
  BUILD_ATTR(type.isSignedInteger(16), int16_t, tensor)
  BUILD_ATTR(type.isSignedInteger(32), int32_t, tensor)
  BUILD_ATTR(type.isSignedInteger(64), int64_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(8), uint8_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(16), uint16_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(32), uint32_t, tensor)
  BUILD_ATTR(type.isSignlessInteger(64), uint64_t, tensor)
  BUILD_ATTR(type.isF32(), float, tensor)
  BUILD_ATTR(type.isF64(), double, tensor)
  UNIMPLEMENTED(llc::MLIR);
  return {};
}

#undef BUILD_ATTR
// mlir::ex::Layout getLayoutFrom(mlir::RankedTensorType tensor) {
//   auto encode =
//       mlir::cast_or_null<mlir::ex::EncodingAttr>(tensor.getEncoding());
//   CHECK(UTILITY, encode) << "tensor not have mlir::ex::EncodingAttr";
//   return encode.getLayout();
// }

// mlir::RankedTensorType cloneTensorWithEncoding(
//     mlir::RankedTensorType tensor, mlir::ex::Layout layout) {
//   auto type = tensor.getElementType();
//   auto shape = tensor.getShape();
//   auto encode = mlir::ex::EncodingAttr::get(tensor.getContext(), layout);
//   return mlir::RankedTensorType::get(shape, type, encode);
// }

std::vector<int64_t> getUnsqueezeShape(mlir::ShapedType shapeType, int dim) {
  auto shape = shapeType.getShape().vec();
  shape.insert(shape.begin() + dim, 1);
  return shape;
}

std::vector<int64_t> getSqueezeShape(mlir::ShapedType shapeType, int dim) {
  auto shape = shapeType.getShape().vec();
  CHECK_GT(UTILITY, shape.size(), 0);
  shape.erase(shape.begin() + dim);
  return shape;
}
mlir::RankedTensorType getUnsqueezeTensor(mlir::RankedTensorType tensor,
                                          int dim) {
  // auto encode =
  //     mlir::ex::EncodingAttr::get(tensor.getContext(),
  //     getLayoutFrom(tensor));
  return mlir::RankedTensorType::get(getUnsqueezeShape(tensor),
                                     tensor.getElementType());
}
mlir::RankedTensorType getSqueezeTensor(mlir::RankedTensorType tensor,
                                        int dim) {
  // auto encode =
  //     mlir::ex::EncodingAttr::get(tensor.getContext(),
  //     getLayoutFrom(tensor));
  return mlir::RankedTensorType::get(getSqueezeShape(tensor),
                                     tensor.getElementType());
}

}  // namespace llc
