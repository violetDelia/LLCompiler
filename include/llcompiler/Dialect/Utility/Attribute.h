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

#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace llc {

enum LAYOUT : int64_t { NCHW = 0, NHWC = 1 };
const char* layout_to_str(LAYOUT layout);

extern const char* OperationNmaeAttr;
extern const char* GloabalLayoutAttr;
extern const char* LayoutAttr;
extern const char* GroupAttr;
extern const char* KernelShapeAttr;
extern const char* IsWeightAttr;
extern const char* PadAttr;
extern const char* SymbolGeneratedAttr;
extern const char* StopRun;

void add_op_name_attr(mlir::Operation* op, llvm::StringRef value);
void add_gloabal_layout_attr(mlir::Operation* op, LAYOUT value);
void add_group_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value);
void add_kernal_shape_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value);
void add_is_weight_attr(mlir::Operation* op, bool value);
void add_layout_attr(mlir::Operation* op, LAYOUT value);
void add_symbol_generate_attr(mlir::Operation* op);
void add_stop_run_attr(mlir::Operation* op);
}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_ATTRIBUTE_H_
