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
#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_TYPE_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_TYPE_H_
#include <cstdint>
#include <vector>

#include "llcompiler/Dialect/IRExtension/IR/Enums.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"


namespace llc {
std::vector<int64_t> getShapeFrom(const mlir::Type& shape_type);
int64_t getElementSizeFrom(const mlir::ShapedType& shape_type);
mlir::ex::Layout getLayoutFrom(const mlir::Value& value);
}  // namespace llc
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_TYPE_H_
