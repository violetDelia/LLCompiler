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

#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_

#define LLC_EMITERROR(condistion) \
  CHECK(llc::MLIR, condistion);   \
  if (condistion) return emitOpError()

#define DEBUG_BUILDED_OP(module, op) \
  DEBUG(module) << "create a " << op.getOperationName().str() << "op.";

#define CONVERT_TO_NEW_OP(module, op, new_op)                          \
  rewriter.replaceOp(op, new_op);                                      \
  DEBUG(module) << "convert " << op.getOperationName().str() << " to " \
                << new_op.getOperationName().str() << "."

#define LLC_COMMON_TYPE_CASE(MACRO, type, ...)             \
  MACRO(type.isInteger(1), bool, __VA_ARGS__)              \
  MACRO(type.isSignedInteger(8), int8_t, __VA_ARGS__)      \
  MACRO(type.isSignedInteger(16), int16_t, __VA_ARGS__)    \
  MACRO(type.isSignedInteger(32), int32_t, __VA_ARGS__)    \
  MACRO(type.isSignedInteger(64), int64_t, __VA_ARGS__)    \
  MACRO(type.isSignlessInteger(8), uint8_t, __VA_ARGS__)   \
  MACRO(type.isSignlessInteger(16), uint16_t, __VA_ARGS__) \
  MACRO(type.isSignlessInteger(32), uint32_t, __VA_ARGS__) \
  MACRO(type.isSignlessInteger(64), uint64_t, __VA_ARGS__) \
  MACRO(type.isF32(), float, __VA_ARGS__)                  \
  MACRO(type.isF64(), double, __VA_ARGS__)
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_
