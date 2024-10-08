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

#include <iostream>
#include <string>

#include "llcompiler/Compiler/Entrance.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace llc::compiler {

void load_dialect(mlir::MLIRContext& context) {
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.getOrLoadDialect<mlir::llh::LLHDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
}

void add_extension_and_interface(mlir::DialectRegistry& registry) {
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
}

void init_logger(const logger::LoggerOption& logger_option) {
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
               << logger::log_level_to_str(logger_option.level);
}

void init_frontend(const front::FrontEndOption& front_option,
                   const logger::LoggerOption& logger_option) {
  logger::register_logger(IMPORTER, logger_option);
  INFO(GLOBAL) << "frontend type is: "
               << front::frontend_type_to_str(front_option.frontend_type);
  INFO(GLOBAL) << "input file is: " << front_option.input_file;
  INFO(GLOBAL) << "output file is: " << front_option.output_file;
  INFO(GLOBAL) << "convert onnx: " << front_option.onnx_convert;
  if (front_option.onnx_convert) {
    INFO(GLOBAL) << "convert onnx to: " << front_option.onnx_convert_version;
  }
}

void generatePiplineOptions(
    CompilerOptions& options,
    llc::pipleline::BasicPipelineOptions& pipleline_options) {
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
  pipleline_options.runMode = str_to_mode(options.mode.c_str());
  pipleline_options.target = str_to_target(options.target.c_str());
  pipleline_options.symbolInfer = options.symbol_infer;
  pipleline_options.irTreeDir = options.ir_tree_dir;
  pipleline_options.indexBitWidth = options.index_bit_width;
}
}  // namespace llc::compiler
