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
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace llc {

enum LAYOUT : int64_t { NCHW = 0, NHWC = 1 };

extern const char* LLCOperationNmaeAttr;
extern const char* LLCLayoutAttr;
extern const char* LLCGroupAttr;
extern const char* LLCKernelShapeAttr;
extern const char* LLCIsWeightAttr;

void add_op_name_attr(mlir::Operation* op, llvm::StringRef value);
void add_layout_attr(mlir::Operation* op, mlir::ArrayRef<LAYOUT> value);
void add_group_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value);
void add_kernal_shape_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value);
void add_is_weight_attr(mlir::Operation* op, bool value);

}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_ATTRIBUTE_H_
