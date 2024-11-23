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

#include "llcompiler/Dialect/TosaExtension/IR/TosaExDialect.h"
#include "llcompiler/Dialect/TosaExtension/IR/TosaExOps.h"
#include "llcompiler/Dialect/TosaExtension/IR/TosaExTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#define FIX_HEADER
#include "llcompiler/Dialect/TosaExtension/IR/TosaExDialect.cpp.inc"
#undef FIX_HEADER

namespace mlir::tosa_ex {
//===----------------------------------------------------------------------===//
// TosaExDialect initialize method.
//===----------------------------------------------------------------------===//
void TosaExDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "llcompiler/Dialect/TosaExtension/IR/TosaExOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// TosaExDialect initialize method.
//===----------------------------------------------------------------------===//

}  // namespace mlir::tosa_ex
