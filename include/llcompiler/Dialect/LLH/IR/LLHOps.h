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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_

#include "llcompiler/Dialect/LLH/IR/LLHTypesImpl.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define PLACEHOLD_FOR_FIX_HEADER
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h.inc"
#include "llcompiler/Dialect/LLH/IR/LLHEunms.h.inc"
#define GET_ATTRDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h.inc"
#define GET_OP_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHOps.h.inc"
#undef PLACEHOLD_FOR_FIX_HEADER
#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHOPS_H_
