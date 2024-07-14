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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHTYPES_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHTYPES_H_

#include "mlir/IR/BuiltinTypes.h"

namespace llc::llh {
enum SIGNED_TAG : uint32_t {
  SIGNLESS,
  SIGNED,
  UNSIGNED,
};
}

#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h.inc"

#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHTYPES_H_
