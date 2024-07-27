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

#include "mlir/IR/Builders.h"

#define LLC_LOG_BUILDED_OP(OP) \
  DEBUG(IMPORTER) << "create a " << OP::getOperationName().str() << "op.";

#define LLC_BUILD_OP(builder, Op, ...) \
  builder.create<::mlir::Op>(builder.getUnknownLoc(), __VA_ARGS__)
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_MACRO_H_
