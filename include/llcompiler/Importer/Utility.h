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
#ifndef INCLUDE_LLCOMPILER_IMPORTER_UTILITY_H_
#define INCLUDE_LLCOMPILER_IMPORTER_UTILITY_H_
/**
 * @file Utility.h
 * @brief utility function about import
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */
#include <any>
#include <string>

#include "llcompiler/Importer/Importer.h"
#include "llcompiler/Support/Option.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace llc::importer {
ImporterOption get_importer_option();

mlir::OwningOpRef<mlir::ModuleOp> gen_mlir_from_to(
    mlir::MLIRContext *context, const ImporterOption &option);
}  // namespace llc::importer

#endif  // INCLUDE_LLCOMPILER_IMPORTER_UTILITY_H_
