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
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Compiler/Utility.h"
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/Utility/File.h"
#include "llcompiler/Frontend/Core/Option.h"
#include "llcompiler/Support/Option.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "llcompiler");
  auto logger_option = llc::option::get_logger_option();
  auto front_option = llc::option::get_front_end_option();
  llc::compiler::init_global(logger_option);
  llc::compiler::init_frontend(front_option, logger_option);
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::llh::LLHDialect>();
  auto module = llc::compiler::gen_mlir_from(&context, front_option);
  mlir::registerConvertLLHToTosa();
  llc::file::mlir_to_file(&module, front_option.output_file.c_str());
  mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Minimal Standalone optimizer driver\n", registry));

  return 0;
}
