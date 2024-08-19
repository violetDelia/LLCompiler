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

#include "llcompiler/Dialect/Utility/Tool.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

namespace llc {
mlir::DenseElementsAttr genDenseElementsFromArrayAttr(
    mlir::DenseI64ArrayAttr attr) {
  auto size = attr.getSize();
  auto element_type = attr.getElementType();
  mlir::SmallVector<int64_t> shape;
  shape.push_back(size);
  auto tensor = mlir::RankedTensorType::get(shape, element_type);
  return mlir::DenseElementsAttr::get(tensor, attr.asArrayRef());
}

}  // namespace llc