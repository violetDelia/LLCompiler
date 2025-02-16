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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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

mlir::ShapedType getShapeTypeFrom(mlir::Type type) {
  CHECK(::llc::UTILITY, mlir::isa<mlir::ShapedType>(type));
  return mlir::cast<mlir::ShapedType>(type);
}

mlir::ShapedType getShapeTypeFrom(mlir::Value value) {
  auto type = value.getType();
  return getShapeTypeFrom(type);
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
}
// one dim shape
mlir::DenseIntElementsAttr ArrayAttrToIntElementsAttr(
    mlir::DenseI64ArrayAttr array_attr) {
  auto data = array_attr.asArrayRef();
  auto shape = llvm::SmallVector<int64_t>();
  auto ele_type = array_attr.getElementType();
  shape.push_back(data.size());
  auto tensor = mlir::RankedTensorType::get(shape, ele_type);
  return mlir::DenseIntElementsAttr::get(tensor, data);
}

mlir::DenseIntElementsAttr GenWindowIntElementsAttr(
    mlir::DenseI64ArrayAttr array_attr, mlir::llh::LayoutAttr layout) {
  auto data = array_attr.asArrayRef();
  auto new_data = layout.addBatchAndFeature(data);
  auto shape = llvm::SmallVector<int64_t>();
  auto ele_type = array_attr.getElementType();
  shape.push_back(data.size() + 2);
  auto tensor = mlir::RankedTensorType::get(shape, ele_type);
  return mlir::DenseIntElementsAttr::get(tensor, new_data);
}

mlir::DenseIntElementsAttr GenWindowPadIntElementsAttr(
    mlir::DenseI64ArrayAttr pad_attr) {
  auto data = pad_attr.asArrayRef();
  int64_t rank = data.size();
  llvm::SmallVector<int64_t> new_data(rank, 0);
  new_data.append(data.begin(), data.end());
  llvm::SmallVector<int64_t> shape = {rank, 2};
  auto ele_type = pad_attr.getElementType();
  auto tensor = mlir::RankedTensorType::get(shape, ele_type);
  return mlir::DenseIntElementsAttr::get(tensor, new_data);
}

#define BUILD_ATTR(judge, Ty, shape, val)                       \
  if (judge) {                                                  \
    llvm::ArrayRef<Ty> value_arr(val);                          \
    auto attr = mlir::DenseElementsAttr::get(shape, value_arr); \
    return attr;                                                \
  }

mlir::DenseElementsAttr genSplatElementAttr(llvm::ArrayRef<int64_t> shape,
                                            mlir::Type element_type,
                                            double value) {
  auto tensor = mlir::RankedTensorType::get(shape, element_type);
  BUILD_ATTR(element_type.isInteger(1), bool, tensor, value)
  BUILD_ATTR(element_type.isSignedInteger(8), int8_t, tensor, value)
  BUILD_ATTR(element_type.isSignedInteger(16), int16_t, tensor, value)
  BUILD_ATTR(element_type.isSignedInteger(32), int32_t, tensor, value)
  BUILD_ATTR(element_type.isSignedInteger(64), int64_t, tensor, value)
  BUILD_ATTR(element_type.isSignlessInteger(8), uint8_t, tensor, value)
  BUILD_ATTR(element_type.isSignlessInteger(16), uint16_t, tensor, value)
  BUILD_ATTR(element_type.isSignlessInteger(32), uint32_t, tensor, value)
  BUILD_ATTR(element_type.isSignlessInteger(64), uint64_t, tensor, value)
  BUILD_ATTR(element_type.isF32(), float, tensor, value)
  BUILD_ATTR(element_type.isF64(), double, tensor, value)
  UNIMPLEMENTED(llc::MLIR);
}
#undef BUILD_ATTR

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
  return mlir::RankedTensorType::get(getUnsqueezeShape(tensor),
                                     tensor.getElementType());
}
mlir::RankedTensorType getSqueezeTensor(mlir::RankedTensorType tensor,
                                        int dim) {
  return mlir::RankedTensorType::get(getSqueezeShape(tensor),
                                     tensor.getElementType());
}

}  // namespace llc
