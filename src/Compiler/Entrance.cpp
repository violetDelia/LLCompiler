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
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

namespace llc::compiler {
void do_compile(const char* xdsl_module, const char* mode, const char* target,
                const char* ir_tree_dir, const char* log_root,
                const char* log_level) {
  // ********* init logger *********//
  logger::LoggerOption logger_option;
  logger_option.level = logger::str_to_log_level(log_level);
  logger_option.path = log_root;
  init_logger(logger_option);
  // ********* init mlir context *********//
  mlir::DialectRegistry registry;
  add_extension_and_interface(registry);
  mlir::MLIRContext context(registry);
  load_dialect(context);
  // ********* load to mlir *********//
  mlir::OwningOpRef<mlir::ModuleOp> module;
  file::str_to_mlir_module(context, module, xdsl_module);
  module->dump();
  // ********* init pipeline options *********//
  pipleline::BasicPipelineOptions pipleline_options;
  pipleline_options.runMode = str_to_mode(mode);
  pipleline_options.target = str_to_target(target);
  pipleline_options.onlyCompiler = false;
  pipleline_options.irTreeDir = ir_tree_dir;
  // ********* process in mlir *********//
  mlir::PassManager pm(module.get()->getName());
  pipleline::buildBasicPipeline(pm, pipleline_options);
  CHECK(MLIR, mlir::succeeded(pm.run(*module))) << "Failed to run pipeline";
  return;
}

}  // namespace llc::compiler
