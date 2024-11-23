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
#include "Dialect/IRExtension/IR/Types.h"

#include "Dialect/IRExtension/IR/Dialect.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/IRExtension/IR/Types.cpp.inc"
namespace mlir::ex {
//===----------------------------------------------------------------------===//
// LLHDialect initialize method.
//===----------------------------------------------------------------------===//
void IRExtensionDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/IRExtension/IR/Types.cpp.inc"
      >();
}
}  // namespace mlir::ex
