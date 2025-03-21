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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHATTRS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHATTRS_H_
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#define GET_ATTRDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h.inc"
namespace mlir::llh {}

#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHATTRS_H_
