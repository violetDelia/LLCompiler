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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"

#include "llcompiler/Dialect/Utility/Macro.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHOps.cpp.inc"

namespace mlir::llh {

#define CHECK_STMBOL(name)                                             \
  auto name = (*this)->getAttrOfType<SymbolRefAttr>(#name);            \
  LLC_EMITERROR(name);                                                 \
  SymbolicIntOp name##_root =                                          \
      symbolTable.lookupNearestSymbolFrom<SymbolicIntOp>(*this, name); \
  LLC_EMITERROR(name##_root);

// LogicalResult SymbolRelationOp::verifySymbolUses(
//     SymbolTableCollection &symbolTable) {
//   CHECK_STMBOL(symbol)
//   CHECK_STMBOL(relation)
//   return llvm::success();
// }

// LogicalResult SymbolBinaryRelationOp::verifySymbolUses(
//     SymbolTableCollection &symbolTable) {
//   CHECK_STMBOL(symbol)
//   CHECK_STMBOL(relations_lhs)
//   CHECK_STMBOL(relations_rhs)
//   return llvm::success();
// }

#undef CHECK_STMBOL
}  // namespace mlir::llh
