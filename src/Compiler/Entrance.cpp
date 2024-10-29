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

#include "llcompiler/Compiler/Engine.h"
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Dialect/Utility/File.h"
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace llc::compiler {

CompilerOptions::CompilerOptions() {}

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
  if (options.log_llvm && options.log_root != "") {
    file::llvm_module_to_file(llvm_module.get(),
                              (options.log_root + "/original.ll").c_str());
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
  if (options.log_llvm && options.log_root != "") {
    file::llvm_module_to_file(llvm_module.get(),
                              (options.log_root + "/final.ll").c_str());
  }

  // ********* link *********//
  // TODO: refine
  llvm::SmallVector<llvm::SmallString<256>, 4> sharedLibPaths;
  sharedLibPaths.push_back(llvm::StringRef(
      "/home/lfr/LLCompiler/build/third_party/llvm-project/llvm/lib/"
      "libmlir_c_runner_utils.so"));
  llvm::StringMap<void*> exportSymbols;
  llvm::SmallVector<mlir::ExecutionEngine::LibraryDestroyFn> destroyFns;
  llvm::SmallVector<llvm::StringRef> jitDyLibPaths;
  for (auto& libPath : sharedLibPaths) {
    auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(
        libPath.str().str().c_str());
    void* initSym =
        lib.getAddressOfSymbol(mlir::ExecutionEngine::kLibraryInitFnName);
    void* destroySim =
        lib.getAddressOfSymbol(mlir::ExecutionEngine::kLibraryDestroyFnName);
    if (!initSym || !destroySim) {
      jitDyLibPaths.push_back(libPath);
      continue;
    }
    auto initFn =
        reinterpret_cast<mlir::ExecutionEngine::LibraryInitFn>(initSym);
    initFn(exportSymbols);
    auto destroyFn =
        reinterpret_cast<mlir::ExecutionEngine::LibraryDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);
  }
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession& session,
                                       const llvm::Triple& tt) {
    auto objectLayer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        session, [sectionMemoryMapper = nullptr]() {
          return std::make_unique<llvm::SectionMemoryManager>(
              sectionMemoryMapper);
        });
    llvm::Triple targetTriple(llvm::Twine(llvm_module->getTargetTriple()));
    if (targetTriple.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    for (auto& libPath : jitDyLibPaths) {
      auto mb = llvm::MemoryBuffer::getFile(libPath);
      if (!mb) {
        FATAL(llc::GLOBAL) << "Failed to create MemoryBuffer for: "
                           << libPath.str()
                           << "\nError: " << mb.getError().message() << "\n";
        continue;
      }
      auto& jd = session.createBareJITDylib(std::string(libPath));
      auto loaded = llvm::orc::DynamicLibrarySearchGenerator::Load(
          libPath.str().c_str(),
          llvm_module->getDataLayout().getGlobalPrefix());
      if (!loaded) {
        FATAL(llc::GLOBAL) << "Could not load " << libPath.str() << ":\n  "
                           << "\n";
        continue;
      }
      jd.addGenerator(std::move(*loaded));
      cantFail(objectLayer->add(jd, std::move(mb.get())));
    }
    return objectLayer;
  };
  // ********* engine *********//
  // TODO: AOT and preload
  auto maybe_jit = llvm::orc::LLJITBuilder()
                       .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                       .setNumCompileThreads(8)
                       .create();
  CHECK(llc::GLOBAL, maybe_jit) << "Failed to create JIT";
  auto& jit = maybe_jit.get();
  auto error = jit->addIRModule(llvm::orc::ThreadSafeModule(
      std::move(llvm_module), std::move(llvm_context)));
  CHECK(llc::GLOBAL, !error) << "Failed to add module!";
  auto maybe_func = jit->lookup("main");
  CHECK(llc::GLOBAL, maybe_func) << "count not find function!";
  Engine engine(std::move(jit));
  return engine;
}

}  // namespace llc::compiler
