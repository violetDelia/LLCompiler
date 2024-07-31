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

#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
namespace llc {
const char* LLCOperationNmaeAttr = "op_name";
const char* LLCLayoutAttr = "layout";
}  // namespace llc

namespace llc {

void add_string_attr(mlir::Operation* op, llvm::StringRef attr_name,
                     llvm::StringRef value) {
  op->setAttr(attr_name, mlir::StringAttr::get(op->getContext(), value));
}

void add_op_name_attr(mlir::Operation* op, llvm::StringRef value) {
  add_string_attr(op, LLCOperationNmaeAttr, value);
}

void add_lay_out_attr(mlir::Operation* op, llvm::StringRef value) {
  CHECK(UTILITY, (value == "NCHW" || value == "NHWC"))
      << "Invalid layout attribute!";
  add_string_attr(op, LLCLayoutAttr, value);
}
}  // namespace llc
