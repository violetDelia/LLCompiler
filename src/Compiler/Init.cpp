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
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "llcompiler/Support/Logger.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace llc::compiler {

void load_dialect(mlir::MLIRContext& context) {
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.getOrLoadDialect<mlir::llh::LLHDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::mhlo::MhloDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context.getOrLoadDialect<mlir::transform::TransformDialect>();
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
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::func::registerTransformDialectExtension(registry);
  
  mlir::affine::registerTransformDialectExtension(registry);
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::memref::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::transform::registerDebugExtension(registry);
  mlir::transform::registerIRDLExtension(registry);
  mlir::transform::registerLoopExtension(registry);
  mlir::transform::registerPDLExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);

  mlir::llh::registerMarkAotPass();
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

}  // namespace llc::compiler
