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
#include "llcompiler/Compiler/Entrance.h"

#include <cassert>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Dialect/Utility/File.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
namespace llc::compiler {
void do_compile(const char* xdsl_module, const char* log_root,
                const char* log_level) {
  logger::LoggerOption logger_option;
  logger_option.level = logger::str_to_log_level(log_level);
  logger_option.path = log_root;
  init_logger(logger_option);
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  load_dialect(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  add_extension_and_interface(registry);
  file::str_to_mlir_module(context, module, xdsl_module);
  module->dump();
  return;
}

}  // namespace llc::compiler