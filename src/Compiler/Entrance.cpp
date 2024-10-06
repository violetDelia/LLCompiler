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

#include <stdio.h>
#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <filesystem>
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
#include "llcompiler/Support/Logger.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace llc::compiler {

CompilerOptions::CompilerOptions(){};

Tensor::Tensor(void* data, void* base, size_t offset, std::vector<size_t>& size,
               std::vector<size_t>& stride)
    : data(data), base(base), offset(offset), size(size), stride(stride) {}
Tensor::Tensor(){};

void Tensor::print() {
  std::cout << "data: " << data << std::endl;
  std::cout << "base: " << base << std::endl;
  std::cout << "offset: " << offset << std::endl;
  std::cout << "size: ";
  for (auto s : size) {
    std::cout << " " << s;
  }
  std::cout << std::endl;
  std::cout << "size: ";
  for (auto s : stride) {
    std::cout << " " << s;
  }
  std::cout << std::endl;
  float* data = reinterpret_cast<float*>(data);
  std::cout << "size: ";
  for (int i = 0; i < 6; i++) {
    std::cout << " " << data[i];
  }
  std::cout << std::endl;
}

Engine::Engine(llvm::orc::LLJIT* engine) : engine(engine){};

void Engine::debug_info() { DINFO << engine; }

Engine do_compile(const char* xdsl_module, CompilerOptions options) {
  // ********* init logger *********//
  logger::LoggerOption logger_option;
  logger_option.level = logger::str_to_log_level(options.log_level.c_str());
  logger_option.path = options.log_root;
  init_logger(logger_option);
  INFO(llc::Entrance_Module) << "\n" << xdsl_module;
  // ********* init mlir context *********//
  mlir::DialectRegistry registry;
  add_extension_and_interface(registry);
  mlir::MLIRContext context(registry);
  load_dialect(context);
  // ********* load to mlir *********//
  mlir::OwningOpRef<mlir::ModuleOp> module;
  file::str_to_mlir_module(context, module, xdsl_module);
  // ********* init pipeline options *********//
  pipleline::BasicPipelineOptions pipleline_options;
  generatePiplineOptions(options, pipleline_options);
  // ********* process in mlir *********//
  mlir::PassManager pm(module.get()->getName());
  if (std::filesystem::exists(pipleline_options.irTreeDir.getValue())) {
    INFO(GLOBAL) << "mlir ir tree dir is: "
                 << pipleline_options.irTreeDir.getValue();
    pm.getContext()->disableMultithreading();
    pm.enableIRPrintingToFileTree(
        [](mlir::Pass* pass, mlir::Operation*) {
          if (pass->getName() == "Operationlegalization") return true;
          return false;
        },
        [](mlir::Pass* pass, mlir::Operation*) { return true; }, false, false,
        false, pipleline_options.irTreeDir, mlir::OpPrintingFlags());
  }
  pipleline::buildBasicPipeline(pm, pipleline_options);
  CHECK(MLIR, mlir::succeeded(pm.run(*module))) << "Failed to run pipeline";
  // ********* convert to llvm ir *********//
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = mlir::translateModuleToLLVMIR(module.get(), *llvm_context);
  CHECK(llc::GLOBAL, llvm_module) << "Failed to emit LLVM IR\n";
  if (options.log_llvm) {
    if (options.log_root == "") {
      WARN(llc::GLOBAL) << " could not find log root!";
    } else {
      auto module_file = options.log_root + "/original.ll";
      DINFO << module_file;
      std::error_code ec;
      llvm::raw_fd_ostream outs(module_file, ec);
      llvm_module->print(outs, nullptr);
      outs.close();
    }
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  CHECK(llc::GLOBAL, tmBuilderOrError)
      << "Could not create JITTargetMachineBuilder\n";
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  CHECK(llc::GLOBAL, tmOrError) << "Could not create TargetMachine\n";
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvm_module.get(),
                                                        tmOrError.get().get());
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/options.opt_level, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  CHECK(llc::GLOBAL, !optPipeline(llvm_module.get()))
      << "Failed to optimize LLVM IR "
      << "\n";
  if (options.log_llvm) {
    if (options.log_root == "") {
      WARN(llc::GLOBAL) << " could not find log root!";
    } else {
      auto module_file = options.log_root + "/final.ll";
      std::error_code ec;
      llvm::raw_fd_ostream outs(module_file, ec);
      llvm_module->print(outs, nullptr);
      outs.close();
    }
  }
  // ********* engine *********//
  auto maybe_jit = llvm::orc::LLJITBuilder().setNumCompileThreads(8).create();
  CHECK(llc::GLOBAL, maybe_jit) << "Failed to create JIT";
  auto& jit = maybe_jit.get();
  auto error = jit->addIRModule(llvm::orc::ThreadSafeModule(
      std::move(llvm_module), std::move(llvm_context)));
  CHECK(llc::GLOBAL, !error) << "Failed to add module!";
  // auto maybe_func = jit->lookup("main");
  // CHECK(llc::GLOBAL, maybe_func) << "count not find function!";
  // auto& func = maybe_func.get();
  // auto run = func.toPtr<StridedMemRefType<float, 3>()>();
  // auto c = run();
  Engine engine(jit.release());
  return engine;
}

}  // namespace llc::compiler
