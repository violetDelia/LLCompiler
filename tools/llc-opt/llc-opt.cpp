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
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/BufferizationExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h"
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/TosaExtension/IR/TosaExDialect.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llh::LLHDialect>();
  registry.insert<mlir::ex::IRExtensionDialect>();
  registry.insert<mlir::tosa_ex::TosaExDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(registry);

  mlir::llh::registerLLHOptPasses();
  mlir::llh::registerLLCSymbolOptPasses();
  mlir::registerLLCConversionPasses();
  mlir::index::ex::registerIndexExtensionPasses();
  mlir::LLVM::ex::registerLLVMExtensionPasses();
  mlir::bufferization::ex::registerBufferizationExtensionPasses();
  mlir::stablehlo::registerAllDialects(registry);
  llc::pipeline::registerTransformPipeline();
  mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "llc-opt", registry));
  return 0;
}
