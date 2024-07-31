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
#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_ATTRIBUTE_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_ATTRIBUTE_H_
#include <utility>

#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"

namespace llc {
extern const char* LLCOperationNmaeAttr;
extern const char* LLCLayoutAttr;
;
}  // namespace llc

namespace llc {

void add_string_attr(mlir::Operation* op, llvm::StringRef attr_name,
                     llvm::StringRef value);

void add_op_name_attr(mlir::Operation* op, llvm::StringRef value);

void add_lay_out_attr(mlir::Operation* op, llvm::StringRef value);
// void add_op_name_attr(mlir::Operation* op, std::string name);

#define ADD_ATTR(key, call)                       \
  if (key_attr == key) {                          \
    return call(op, std::forward<Args>(args)...); \
  }

template <class... Args>
void add_attr(mlir::Operation* op, const char* key_attr, Args... args) {
  ADD_ATTR(LLCOperationNmaeAttr, add_op_name_attr)
  ADD_ATTR(LLCLayoutAttr, add_lay_out_attr)
  UNIMPLEMENTED(UTILITY);
}
#undef ADD_ATTR
}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_ATTRIBUTE_H_
