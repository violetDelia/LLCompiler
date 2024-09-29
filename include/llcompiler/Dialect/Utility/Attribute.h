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

#define DEF_ATTR(name, Key) extern const char* Key;

#define ADD_STRING_ATTR(name, Key)                                    \
  DEF_ATTR(name, Key)                                                 \
  void add_##name##_attr(mlir::Operation* op, llvm::StringRef value); \
  void add_##name##_attr(mlir::Operation* op, const char* value);

#define ADD_DENSE_I64_ATTR(name, Key) \
  DEF_ATTR(name, Key)                 \
  void add_##name##_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value);

#define ADD_UNIT_ATTR(name, Key) \
  DEF_ATTR(name, Key)            \
  void add_##name##_attr(mlir::Operation* op);

#define ADD_LAYOUT_ATTR(name, Key) \
  DEF_ATTR(name, Key)              \
  void add_##name##_attr(mlir::Operation* op, LAYOUT value);

ADD_LAYOUT_ATTR(gloabal_layout, GloabalLayoutAttr)
ADD_LAYOUT_ATTR(layout, LayoutAttr)
ADD_STRING_ATTR(op_name, OperationNameAttr)
ADD_DENSE_I64_ATTR(group, GroupAttr)
ADD_DENSE_I64_ATTR(kernal_shape, KernelShapeAttr)
ADD_DENSE_I64_ATTR(stride, StrideAtt)
ADD_DENSE_I64_ATTR(pad, PadAttr)
ADD_UNIT_ATTR(is_weight, IsWeightAttr)
ADD_UNIT_ATTR(symbol_generate, SymbolGeneratedAttr)
ADD_UNIT_ATTR(stop_run, StopRun)
ADD_UNIT_ATTR(entrance, Entrance)

#undef ADD_LAYOUT_ATTR
#undef ADD_STRING_ATTR
#undef ADD_DENSE_I64_ATTR
#undef ADD_UNIT_ATTR
#undef DEF_ATTR

}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_ATTRIBUTE_H_
