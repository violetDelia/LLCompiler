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
#include <any>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace llc {

llvm::ArrayRef<int64_t> get_shape_form(const mlir::Type& shape_type) {
  CHECK(UTILITY, mlir::isa<mlir::ShapedType>(shape_type));
  return mlir::cast<mlir::ShapedType>(shape_type).getShape();
}

int64_t get_element_size_form(const mlir::ShapedType& shape_type) {
  auto rank = shape_type.getRank();
  if (rank == 0) return 0;
  int64_t element_size = 1;
  for (size_t i = 0; i < rank; ++i) {
    element_size *= shape_type.getDimSize(i);
  }
  return element_size;
}
}  // namespace llc
