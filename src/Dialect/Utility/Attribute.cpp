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
//**

#include <cstdint>

#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

namespace llc {
const char* LLCOperationNmaeAttr = "op_name";
const char* LLCLayoutAttr = "layout";
const char* LLCGroupAttr = "group";
const char* LLCKernelShapeAttr = "kernel_shape";
const char* LLCIsWeightAttr = "is_weight";
}  // namespace llc

namespace llc {
#define ADD_ATTR_FUNC(name, input_type, attr_type)                       \
  void add_##name##_attr(mlir::Operation* op, llvm::StringRef attr_name, \
                         input_type value) {                             \
    op->setAttr(attr_name, attr_type::get(op->getContext(), value));     \
  }

ADD_ATTR_FUNC(array_i64, llvm::SmallVector<int64_t>, mlir::DenseI64ArrayAttr)
ADD_ATTR_FUNC(array_i64, llvm::ArrayRef<int64_t>, mlir::DenseI64ArrayAttr)
ADD_ATTR_FUNC(array_i1, llvm::SmallVector<bool>, mlir::DenseBoolArrayAttr)
ADD_ATTR_FUNC(array_i1, llvm::ArrayRef<bool>, mlir::DenseBoolArrayAttr)
ADD_ATTR_FUNC(bool, bool, mlir::BoolAttr)
ADD_ATTR_FUNC(string, llvm::StringRef, mlir::StringAttr)
ADD_ATTR_FUNC(string, const char*, mlir::StringAttr)
#undef ADD_ATTR_FUNC

void add_op_name_attr(mlir::Operation* op, llvm::StringRef value) {
  add_string_attr(op, LLCOperationNmaeAttr, value);
}
void add_layout_attr(mlir::Operation* op, mlir::ArrayRef<LAYOUT> value) {
  mlir::SmallVector<int64_t> attr_value;
  for (auto layout : value) {
    attr_value.push_back(static_cast<int64_t>(layout));
  }
  add_array_i64_attr(op, LLCLayoutAttr, attr_value);
}
void add_group_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value) {
  add_array_i64_attr(op, LLCGroupAttr, value);
}
void add_kernal_shape_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value) {
  add_array_i64_attr(op, LLCKernelShapeAttr, value);
}
void add_is_weight_attr(mlir::Operation* op, bool value) {
  add_bool_attr(op, LLCIsWeightAttr, value);
}

void add_lay_out_attr(mlir::Operation* op, llvm::StringRef value) {
  CHECK(UTILITY, (value == "NCHW" || value == "NHWC"))
      << "Invalid layout attribute!";
  add_string_attr(op, LLCLayoutAttr, value);
}
}  // namespace llc
