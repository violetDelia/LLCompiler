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

#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Pipeline/CommonPipeline.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Option.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

//--dump-pass-pipeline --inline --convert-llh-to-tosa
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  auto logger_option = llc::option::get_logger_option();
  llc::logger::register_logger(llc::MLIR, logger_option);
  llc::logger::register_logger(llc::UTILITY, logger_option);
  registry.insert<mlir::llh::LLHDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::ex::IRExtensionDialect>();
  mlir::func::registerInlinerExtension(registry);
  llc::pipleline::registerCommonPipeline();
  mlir::llh::registerLLHOptPasses();
  mlir::registerConvertLLHToTosaPass();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "llc-compiler", registry));
  return 0;
}
