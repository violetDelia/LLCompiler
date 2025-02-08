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
 * @file Init.h
 * @brief initializing compiler
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */

#ifndef INCLUDE_LLCOMPILER_COMPILER_COMPILER_H_
#define INCLUDE_LLCOMPILER_COMPILER_COMPILER_H_
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "llcompiler/Compiler/CompileOption.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
namespace llc::compiler {
class LLCCompiler {
 public:
  LLCCompiler();
  std::string generateSharedLibFromMLIRFile(std::string module_file,
                                            CompileOptions options);
  std::string generateSharedLibFromMLIRStr(std::string module_str,
                                           CompileOptions options);

 private:
  void registerLogger(CompileOptions options);

  void optimizeMLIRFile(std::string module_file, CompileOptions options,
                        std::string opted_module_file);

  void optimizeMLIRStr(std::string module_string, CompileOptions options,
                       std::string opted_module_file);
  void optimizeMLIR(mlir::OwningOpRef<mlir::ModuleOp>& module,
                    CompileOptions options, std::string opted_module_file);

  void translateMLIRToLLVMIR(std::string mlir_file, CompileOptions options,
                             std::string llvm_bitcode_file);

  void optimizeLLVMIR(std::string llvm_ir_file, CompileOptions options,
                      std::string opted_llvm_ir_file);

  void translateLLVMIRToBitcode(std::string opted_llvm_ir_file,
                                CompileOptions options,
                                std::string llvm_bitcode_file);

  void translateBitcodeToObject(std::string llvm_bitcode_file,
                                CompileOptions options,
                                std::string object_file);

  void generateSharedLib(std::vector<std::string> objs, CompileOptions options,
                         std::string shared_lib_file);
};

}  // namespace llc::compiler
#endif  // INCLUDE_LLCOMPILER_COMPILER_COMPILER_H_
