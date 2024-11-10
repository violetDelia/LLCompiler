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

#include "deallocation/transforms/passes.h"
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/TosaExtension/IR/TosaExDialect.h"
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Pipeline/CommonPipeline.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo_ext/transforms/passes.h"
#include "transforms/gpu_passes.h"
#include "transforms/passes.h"
//  -pass-pipeline=
//    "builtin.module(  inline,
//                      convert-llh-to-tosa,
//       )"
//
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llh::LLHDialect>();
  registry.insert<mlir::ex::IRExtensionDialect>();
  registry.insert<mlir::tosa_ex::TosaExDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);

  mlir::llh::registerLLHOptPasses();
  mlir::registerLLCConversionPasses();
  mlir::index::ex::registerIndexExtensionPasses();
  mlir::LLVM::ex::registerLLVMExtensionPasses();

  mlir::deallocation::registerDeallocationPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::registerLMHLOGPUTransformsPasses();
  mlir::stablehlo_ext::registerPasses();
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "llc-compiler", registry));
  return 0;
}
