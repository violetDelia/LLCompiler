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

#include "llcompiler/Compiler/Compiler.h"
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/BufferizableOpInterfaceImpl.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/TosaExtension/IR/TosaExDialect.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/reference/Tensor.h"

namespace llc::compiler {

void load_dialect(mlir::MLIRContext& context) {
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  context.getOrLoadDialect<mlir::quant::QuantDialect>();
  context.getOrLoadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  context.getOrLoadDialect<mlir::transform::TransformDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::shape::ShapeDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::complex::ComplexDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();

  context.getOrLoadDialect<mlir::llh::LLHDialect>();
  context.getOrLoadDialect<mlir::chlo::ChloDialect>();
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context.getOrLoadDialect<mlir::vhlo::VhloDialect>();
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
  mlir::shape::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::llh::registerBufferizableOpInterfaceExternalModels(registry);


  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);

  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);

  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);

  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlir::affine::registerTransformDialectExtension(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::bufferization::registerTransformDialectExtension(registry);
  mlir::func::registerTransformDialectExtension(registry);
  mlir::gpu::registerTransformDialectExtension(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::memref::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);

  mlir::transform::registerDebugExtension(registry);
  mlir::transform::registerIRDLExtension(registry);
  mlir::transform::registerLoopExtension(registry);
  mlir::transform::registerPDLExtension(registry);

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
  logger::register_logger(Executor, logger_option);
  INFO(GLOBAL) << "log root is: " << logger_option.path;
  INFO(GLOBAL) << "log level is: "
               << llc::stringifyLogLevel(logger_option.level).str();
}

void preprocess_mlir_module(mlir::OwningOpRef<mlir::ModuleOp>* module,
                            CompileOptions compiler_options) {
  auto context = module->get()->getContext();
  auto maybe_layout = mlir::llh::symbolizeLayout(
      llc::stringifyGlobalLayout(compiler_options.global_layout));
  CHECK(llc::GLOBAL, maybe_layout.has_value())
      << "Layout error: "
      << llc::stringifyGlobalLayout(compiler_options.global_layout).str();
  auto layout = maybe_layout.value();
  module->get()->setAttr(llc::GloabalLayoutAttr,
                         mlir::llh::LayoutAttr::get(context, layout));
  auto maybe_mode = mlir::llh::symbolizeModeKind(
      llc::stringifyModeKind(compiler_options.mode));
  CHECK(llc::GLOBAL, maybe_mode.has_value())
      << "mode error: " << llc::stringifyModeKind(compiler_options.mode).str();
  auto mode = maybe_mode.value();
  module->get()->setAttr(llc::GloabalModeKindAttr,
                         mlir::llh::ModeKindAttr::get(context, mode));
};

}  // namespace llc::compiler
