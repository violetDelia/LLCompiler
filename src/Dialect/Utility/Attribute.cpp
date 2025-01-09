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

#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

namespace llc {
#define ADD_ATTR_FUNC(name, input_type, attr_type)                       \
  void add_##name##_attr(mlir::Operation* op, llvm::StringRef attr_name, \
                         input_type value) {                             \
    auto attr = attr_type::get(op->getContext(), value);                 \
    op->setAttr(attr_name, attr);                                        \
  }

ADD_ATTR_FUNC(array_i64, llvm::SmallVector<int64_t>, mlir::DenseI64ArrayAttr)
ADD_ATTR_FUNC(array_i64, llvm::ArrayRef<int64_t>, mlir::DenseI64ArrayAttr)
ADD_ATTR_FUNC(array_i1, llvm::SmallVector<bool>, mlir::DenseBoolArrayAttr)
ADD_ATTR_FUNC(array_i1, llvm::ArrayRef<bool>, mlir::DenseBoolArrayAttr)
ADD_ATTR_FUNC(bool, bool, mlir::BoolAttr)
ADD_ATTR_FUNC(string, llvm::StringRef, mlir::StringAttr)
ADD_ATTR_FUNC(string, const char*, mlir::StringAttr)
ADD_ATTR_FUNC(flat_symbol, llvm::StringRef, mlir::FlatSymbolRefAttr)
#undef ADD_ATTR_FUNC
void add_unit_attr(mlir::Operation* op, llvm::StringRef attr_name) {
  op->setAttr(attr_name, mlir::UnitAttr::get(op->getContext()));
}

#define DEF_ATTR(name, Key) const char* Key = #name;

#define ADD_STRING_ATTR(name, Key)                                     \
  DEF_ATTR(name, Key)                                                  \
  void add_##name##_attr(mlir::Operation* op, llvm::StringRef value) { \
    add_string_attr(op, Key, value);                                   \
  }                                                                    \
  void add_##name##_attr(mlir::Operation* op, const char* value) {     \
    add_string_attr(op, Key, value);                                   \
  }

#define ADD_SYMBOL_ATTR(name, Key)                                     \
  DEF_ATTR(name, Key)                                                  \
  void add_##name##_attr(mlir::Operation* op, llvm::StringRef value) { \
    add_flat_symbol_attr(op, Key, value);                              \
  }                                                                    \
  void add_##name##_attr(mlir::Operation* op, const char* value) {     \
    add_flat_symbol_attr(op, Key, value);                              \
  }

#define ADD_DENSE_I64_ATTR(name, Key)                                          \
  DEF_ATTR(name, Key)                                                          \
  void add_##name##_attr(mlir::Operation* op, mlir::ArrayRef<int64_t> value) { \
    add_array_i64_attr(op, Key, value);                                        \
  }

#define ADD_UNIT_ATTR(name, Key) \
  DEF_ATTR(name, Key)            \
  void add_##name##_attr(mlir::Operation* op) { add_unit_attr(op, Key); }

#define ADD_LAYOUT_ATTR(name, Key)                                           \
  DEF_ATTR(name, Key)                                                        \
  void add_##name##_attr(mlir::Operation* op, ::mlir::llh::Layout value) {   \
    op->setAttr(Key, ::mlir::llh::LayoutAttr::get(op->getContext(), value)); \
  }

const char* GloabalLayoutAttr = "builtin.gloabal_layout";
const char* GloabalModeKindAttr = "builtin.mode";
const char* FuncSymbolIntAttr = "func.symbol_int";
ADD_LAYOUT_ATTR(layout, LayoutAttr)
ADD_LAYOUT_ATTR(weight_layout, WeightLayoutAttr)
ADD_STRING_ATTR(op_name, OperationNameAttr)
ADD_SYMBOL_ATTR(symbol, SymbolIntAttr)
ADD_DENSE_I64_ATTR(group, GroupAttr)
ADD_DENSE_I64_ATTR(kernel_shape, KernelShapeAttr)
ADD_DENSE_I64_ATTR(stride, StrideAtt)
ADD_DENSE_I64_ATTR(pad, PadAttr)
ADD_DENSE_I64_ATTR(dilation, DilationAttr)
ADD_UNIT_ATTR(is_weight, IsWeightAttr)
ADD_UNIT_ATTR(symbol_generate, SymbolGeneratedAttr)
ADD_UNIT_ATTR(stop_run, StopRunAttr)
ADD_UNIT_ATTR(entrance, EntranceAttr)

#undef ADD_LAYOUT_ATTR
#undef ADD_STRING_ATTR
#undef ADD_DENSE_I64_ATTR
#undef ADD_UNIT_ATTR
#undef DEF_ATTR
}  // namespace llc
