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
#include "llcompiler/Compiler/Compiler.h"

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

#include "llcompiler/Compiler/Command.h"
#include "llcompiler/Compiler/CompileOption.h"
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Dialect/Utility/File.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "mlir/IR/OperationSupport.h"

namespace llc::compiler {

namespace {

void setIRDumpConfig(CompileOptions& options, mlir::PassManager& pm) {
  if (options.log_root.empty()) return;
  if (!options.display_mlir_passes) return;
  if (!std::filesystem::exists(options.log_root))
    std::filesystem::create_directory(options.log_root);
  INFO(GLOBAL) << "mlir ir tree dir is: " << options.log_root;
  pm.enableIRPrintingToFileTree(
      [](mlir::Pass* pass, mlir::Operation*) { return false; },
      [](mlir::Pass* pass, mlir::Operation*) { return true; }, false, false,
      false, options.log_root, mlir::OpPrintingFlags());
}

void generatePipelineOptions(
    CompileOptions& options,
    llc::pipeline::TransformPipelineOptions& pipleline_options) {
  if (options.L1_cache_size == 0) {
    pipleline_options.L1CacheSize = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  } else {
    pipleline_options.L1CacheSize = options.L1_cache_size;
  }
  if (options.L2_cache_size == 0) {
    pipleline_options.L2CacheSize = sysconf(_SC_LEVEL2_CACHE_SIZE);
  } else {
    pipleline_options.L2CacheSize = options.L2_cache_size;
  }
  if (options.L3_cache_size == 0) {
    pipleline_options.L3CacheSize = sysconf(_SC_LEVEL3_CACHE_SIZE);
  } else {
    pipleline_options.L3CacheSize = options.L3_cache_size;
  }
  INFO(llc::GLOBAL) << "L3 Cache Size: "
                    << pipleline_options.L3CacheSize.getValue();
  INFO(llc::GLOBAL) << "L2 Cache Size: "
                    << pipleline_options.L2CacheSize.getValue();
  INFO(llc::GLOBAL) << "L1 Cache Size: "
                    << pipleline_options.L1CacheSize.getValue();
  auto maybe_target_layout = mlir::llh::symbolizeLayout(
      llc::stringifyGlobalLayout(options.global_layout));
  CHECK(llc::GLOBAL, maybe_target_layout.has_value());
  pipleline_options.targetLayout = maybe_target_layout.value();
}

void runTransformPipeline(CompileOptions& options,
                          mlir::OwningOpRef<mlir::ModuleOp>& module) {
  pipeline::TransformPipelineOptions pipleline_options;
  generatePipelineOptions(options, pipleline_options);
  // ********* process in mlir *********//
  mlir::PassManager pm(module.get()->getName());
  setIRDumpConfig(options, pm);
  pipeline::buildTransformPipeline(pm, pipleline_options);
  CHECK(MLIR, mlir::succeeded(pm.run(*module))) << "Failed to run pipeline";
}
}  // namespace

LLCCompiler::LLCCompiler() {}

void LLCCompiler::registerLogger(CompileOptions options) {
  logger::LoggerOption logger_option;
  auto log_dir = llvm::SmallString<128>(options.log_root);
  llvm::sys::path::append(log_dir, "log");
  if (!llvm::sys::fs::exists(log_dir)) {
    INFO(GLOBAL) << "create folder: " << log_dir.c_str();
    llvm::sys::fs::create_directory(log_dir);
  }
  logger_option.path = log_dir.c_str();
  logger_option.level = options.log_level;
  logger::register_logger(GLOBAL, logger_option);
  logger::register_logger(UTILITY, logger_option);
  logger::register_logger(IMPORTER, logger_option);
  logger::register_logger(MLIR, logger_option);
  logger::register_logger(MLIR_PASS, logger_option);
  logger::register_logger(DEBUG, logger_option);
  logger::register_logger(SymbolInfer, logger_option);
  logger::register_logger(Entrance_Module, logger_option);
  INFO(GLOBAL) << "log root is: " << logger_option.path;
  INFO(GLOBAL) << "log level is: "
               << llc::stringifyLogLevel(logger_option.level).str();
};

void LLCCompiler::optimizeMLIRFile(std::string module_file,
                                   CompileOptions options,
                                   std::string opted_module_file) {
  mlir::DialectRegistry registry;
  add_extension_and_interface(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  load_dialect(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  file::file_to_mlir_module(context, module, module_file.c_str());
  optimizeMLIR(module, options, opted_module_file);
};

void LLCCompiler::optimizeMLIR(mlir::OwningOpRef<mlir::ModuleOp>& module,
                               CompileOptions options,
                               std::string opted_module_file) {
  preprocess_mlir_module(&module, options);
  if (options.pipeline == "transform") {
    runTransformPipeline(options, module);
  } else {
    UNIMPLEMENTED(llc::GLOBAL);
  }
  file::mlir_to_file(&module, opted_module_file.c_str());
};

void LLCCompiler::optimizeMLIRStr(std::string module_string,
                                  CompileOptions options,
                                  std::string opted_module_file) {
  mlir::DialectRegistry registry;
  add_extension_and_interface(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  load_dialect(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  file::str_to_mlir_module(context, module, module_string.c_str());
  optimizeMLIR(module, options, opted_module_file);
};

void LLCCompiler::translateMLIRToLLVMIR(std::string mlir_file,
                                        CompileOptions options,
                                        std::string llvm_ir_file) {
  Command translate_mlir(getToolPath("llc-translate"));
  translate_mlir.appendStr(mlir_file)
      .appendStr("-mlir-to-llvmir")
      .appendStr("-allow-unregistered-dialect")
      .appendList({"-o", llvm_ir_file})
      .exec();
}

void LLCCompiler::optimizeLLVMIR(std::string llvm_ir_file,
                                 CompileOptions options,
                                 std::string opted_llvm_ir_file) {
  Command optimize_bitcode(getToolPath("opt"));
  optimize_bitcode.appendStr(llvm_ir_file)
      .appendList({"-o", opted_llvm_ir_file})
      .appendStr(getOptimizationLevelOption(options))
      .appendStr("-S")
      .appendStr(getTargetArchOption(options))
      .appendStr(getCPUOption(options))
      .appendStr(getMtripleOption(options));
  if (options.display_llvm_passes) {
    llvm::SmallString<128> all_passes_file(opted_llvm_ir_file);
    llvm::SmallString<128> llvm_ir_dir =
        llvm::sys::path::parent_path(opted_llvm_ir_file);
    llvm::sys::path::append(llvm_ir_dir, "llvm_dump");
    optimize_bitcode.appendStr("-print-before-all");
    optimize_bitcode.appendList({"-ir-dump-directory", llvm_ir_dir.c_str()});
  }
  optimize_bitcode.exec();
}

void LLCCompiler::translateLLVMIRToBitcode(std::string opted_llvm_ir_file,
                                           CompileOptions options,
                                           std::string llvm_bitcode_file) {
  Command translate_to_bitcode(getToolPath("opt"));
  translate_to_bitcode.appendStr(opted_llvm_ir_file)
      .appendList({"-o", llvm_bitcode_file})
      .appendStr(getTargetArchOption(options))
      .appendStr(getCPUOption(options))
      .appendStr(getMtripleOption(options))
      .exec();
}

void LLCCompiler::translateBitcodeToObject(std::string llvm_bitcode_file,
                                           CompileOptions options,
                                           std::string object_file) {
  Command bitcode_to_object(getToolPath("llc"));
  bitcode_to_object.appendStr(llvm_bitcode_file)
      .appendList({"-o", object_file})
      .appendStr("-filetype=obj")
      .appendStr("-relocation-model=pic")
      .appendStr(getTargetArchOption(options))
      .appendStr(getCPUOption(options))
      .appendStr(getMtripleOption(options))
      .exec();
}

void LLCCompiler::generateSharedLib(std::vector<std::string> objs,
                                    CompileOptions options,
                                    std::string shared_lib_file) {
  Command link(getToolPath("cxx"));
  link.appendList(objs)
      .appendList({"-o", shared_lib_file})
      .appendList({"-shared", "-fPIC"});
  for (auto lib : options.libs) {
    link.appendStr("-l" + lib);
  }
  for (auto lib_dir : options.libdirs) {
    link.appendStr("-L" + lib_dir);
  }
  link.exec();
}

std::string LLCCompiler::generateSharedLibFromMLIRFile(std::string module_file,
                                                       CompileOptions options) {
  registerLogger(options);
  mlir::DialectRegistry registry;
  add_extension_and_interface(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  load_dialect(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  file::file_to_mlir_module(context, module, module_file.c_str());
  llvm::SmallString<4> file_prefix;
  if (options.log_root.empty()) {
    llvm::sys::path::system_temp_directory(true, file_prefix);
    auto hash = std::to_string(llvm::hash_value(module_file));
    llvm::sys::path::append(file_prefix, hash);
  } else {
    file_prefix = options.log_root;
    llvm::sys::path::append(file_prefix, "llc_module");
    auto mlir_file(file_prefix);
    llvm::sys::path::replace_extension(mlir_file, ".mlir");
    file::mlir_to_file(&module, mlir_file.c_str());
  }
  auto mlir_file(file_prefix);
  llvm::sys::path::replace_extension(mlir_file, ".mlir");
  file::mlir_to_file(&module, mlir_file.c_str());
  auto opted_mlir_file(file_prefix);
  llvm::sys::path::replace_extension(opted_mlir_file, ".opted.mlir");
  optimizeMLIR(module, options, opted_mlir_file.str().str());
  auto llvm_ir_file(file_prefix);
  llvm::sys::path::replace_extension(llvm_ir_file, ".ll");
  translateMLIRToLLVMIR(opted_mlir_file.str().str(), options,
                        llvm_ir_file.str().str());
  auto opted_llvm_ir_file(file_prefix);
  llvm::sys::path::replace_extension(opted_llvm_ir_file, "opted.ll");
  optimizeLLVMIR(llvm_ir_file.str().str(), options,
                 opted_llvm_ir_file.str().str());
  auto bitcode_file(file_prefix);
  llvm::sys::path::replace_extension(bitcode_file, ".bc");
  translateLLVMIRToBitcode(opted_llvm_ir_file.str().str(), options,
                           bitcode_file.str().str());
  auto obj_file(file_prefix);
  llvm::sys::path::replace_extension(obj_file, ".o");
  translateBitcodeToObject(bitcode_file.str().str(), options,
                           obj_file.str().str());
  auto shared_lib_file(file_prefix);
  llvm::sys::path::replace_extension(shared_lib_file, ".so");
  generateSharedLib({obj_file.str().str()}, options,
                    shared_lib_file.str().str());
  if (options.log_root.empty()) {
    llvm::FileRemover opted_mlir_remover(opted_mlir_file);
    llvm::FileRemover llvm_ir_remover(llvm_ir_file);
    llvm::FileRemover opted_llvm_ir_remover(opted_llvm_ir_file);
    llvm::FileRemover bitcode_remover(bitcode_file);
    llvm::FileRemover obj_remover(obj_file);
  }
  return shared_lib_file.str().str();
};

std::string LLCCompiler::generateSharedLibFromMLIRStr(std::string module_str,
                                                      CompileOptions options) {
  registerLogger(options);
  mlir::DialectRegistry registry;
  add_extension_and_interface(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  load_dialect(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  file::str_to_mlir_module(context, module, module_str.c_str());
  llvm::SmallString<4> file_prefix;
  if (options.log_root.empty()) {
    llvm::sys::path::system_temp_directory(true, file_prefix);
    auto hash = std::to_string(llvm::hash_value(module_str));
    llvm::sys::path::append(file_prefix, hash);
  } else {
    file_prefix = options.log_root;
    llvm::sys::path::append(file_prefix, "llc_module");
    auto mlir_file(file_prefix);
    llvm::sys::path::replace_extension(mlir_file, ".mlir");
    file::mlir_to_file(&module, mlir_file.c_str());
  }
  auto opted_mlir_file(file_prefix);
  llvm::sys::path::replace_extension(opted_mlir_file, ".opted.mlir");
  optimizeMLIR(module, options, opted_mlir_file.str().str());
  auto llvm_ir_file(file_prefix);
  llvm::sys::path::replace_extension(llvm_ir_file, ".ll");
  translateMLIRToLLVMIR(opted_mlir_file.str().str(), options,
                        llvm_ir_file.str().str());
  auto opted_llvm_ir_file(file_prefix);
  llvm::sys::path::replace_extension(opted_llvm_ir_file, "opted.ll");
  optimizeLLVMIR(llvm_ir_file.str().str(), options,
                 opted_llvm_ir_file.str().str());
  auto bitcode_file(file_prefix);
  llvm::sys::path::replace_extension(bitcode_file, ".bc");
  translateLLVMIRToBitcode(opted_llvm_ir_file.str().str(), options,
                           bitcode_file.str().str());
  auto obj_file(file_prefix);
  llvm::sys::path::replace_extension(obj_file, ".o");
  translateBitcodeToObject(bitcode_file.str().str(), options,
                           obj_file.str().str());
  auto shared_lib_file(file_prefix);
  llvm::sys::path::replace_extension(shared_lib_file, ".so");
  generateSharedLib({obj_file.str().str()}, options,
                    shared_lib_file.str().str());
  if (options.log_root.empty()) {
    llvm::FileRemover opted_mlir_remover(opted_mlir_file);
    llvm::FileRemover llvm_ir_remover(llvm_ir_file);
    llvm::FileRemover opted_llvm_ir_remover(opted_llvm_ir_file);
    llvm::FileRemover bitcode_remover(bitcode_file);
    llvm::FileRemover obj_remover(obj_file);
  }
  return shared_lib_file.str().str();
}

}  // namespace llc::compiler
