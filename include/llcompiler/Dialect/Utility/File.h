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
#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_FILE_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_FILE_H_
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
namespace llc::file {
void mlir_to_file(mlir::OwningOpRef<mlir::ModuleOp>* module, const char* file);
void str_to_mlir_module(mlir::MLIRContext& context,
                        mlir::OwningOpRef<mlir::ModuleOp>& module,
                        const char* str);
void file_to_mlir_module(mlir::MLIRContext& context,
                         mlir::OwningOpRef<mlir::ModuleOp>& module,
                         const char* file);
void llvm_module_to_file(llvm::Module* module, const char* file);
}  // namespace llc::file

#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_FILE_H_
