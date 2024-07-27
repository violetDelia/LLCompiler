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

/**
 * @file Utility.h
 * @brief utility for compiler
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */

#ifndef INCLUDE_LLCOMPILER_COMPILER_UTILITY_H_
#define INCLUDE_LLCOMPILER_COMPILER_UTILITY_H_

#include "llcompiler/Frontend/Core/Base.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/ir/BuiltinOps.h"


namespace llc::compiler {
mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_from(
    mlir::MLIRContext *context, const front::ImporterOption &option);
}

#endif  // INCLUDE_LLCOMPILER_COMPILER_UTILITY_H_
