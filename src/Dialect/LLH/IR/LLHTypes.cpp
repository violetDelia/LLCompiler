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

#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

namespace llc::llh {
//===----------------------------------------------------------------------===//
// LLHDialect initialize method.
//===----------------------------------------------------------------------===//
void LLHDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llcompiler/Dialect/LLH/IR/LLHTypes.cpp.inc"
      >();
}

void printSIGNED_TAG(::mlir::AsmPrinter &printer, SIGNED_TAG tag) {
  switch (tag) {
    case UNSIGNED:
      printer << "u";
    default:
      return;
  }
}

llvm::ParseResult parseSIGNED_TAG(::mlir::AsmParser &parser, SIGNED_TAG &tag) {
  WARN_UNIMPLEMENTED(LLH);
  return mlir::success();
}
//===----------------------------------------------------------------------===//
// LLH type verify
//===----------------------------------------------------------------------===//
::llvm::LogicalResult IntType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned width, SIGNED_TAG signed_tag) {
  if (width > Max_Width) {
    return emitError() << "IntType max bitwidth cant greater than "
                       << Max_Width;
  }
  return llvm::success();
}
}  // namespace llc::llh

#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHTypes.cpp.inc"
