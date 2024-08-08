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
#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

//--dump-pass-pipeline --inline --convert-llh-to-tosa
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::llh::LLHDialect>();
  registry.insert<mlir::tosa ::TosaDialect>();
  mlir::registerAllExtensions(registry);

  mlir::registerTransformsPasses();
  mlir::registerConvertLLHToTosa();
  mlir::tosa::registerTosaOptPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "llc-opt", registry));
}
