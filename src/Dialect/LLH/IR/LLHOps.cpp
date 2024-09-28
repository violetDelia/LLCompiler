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
LogicalResult SymbolRelationsOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto symbol = (*this)->getAttrOfType<SymbolRefAttr>("symbol");
  LLC_EMITERROR(symbol);
  SymbolicIntOp symbol_root =
      symbolTable.lookupNearestSymbolFrom<SymbolicIntOp>(*this, symbol);
  LLC_EMITERROR(symbol_root);
  auto relations = (*this)->getAttrOfType<ArrayAttr>("relations");
  for (auto attr : relations) {
    auto relation_symbol = llvm::cast<StringAttr>(attr);
    LLC_EMITERROR(relation_symbol);
    SymbolicIntOp relation_root =
        symbolTable.lookupNearestSymbolFrom<SymbolicIntOp>(*this,
                                                           relation_symbol);
    LLC_EMITERROR(relation_root);
  }
  return llvm::success();
}
}  // namespace mlir::llh
