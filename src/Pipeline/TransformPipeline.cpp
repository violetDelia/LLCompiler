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
#include "llcompiler/Conversion/LLHToArith/LLHToArith.h"
#include "llcompiler/Conversion/LLHToHLO/LLHToHLO.h"
#include "llcompiler/Conversion/LLHToTensor/LLHToTensor.h"
#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/BufferizationExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/TosaExtension/Transforms/Passes.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/TransformLibrary/LibraryEntry.h"
#include "llcompiler/TransformLibrary/LibraryPath.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
namespace llc::pipeline {
namespace {
void registerPasses() {
  mlir::llh::registerLLHOptPasses();
  mlir::llh::registerLLCSymbolOptPasses();
  mlir::registerLLCConversionPasses();
  mlir::index::ex::registerIndexExtensionPasses();
  mlir::LLVM::ex::registerLLVMExtensionPasses();
  mlir::bufferization::ex::registerBufferizationExtensionPasses();

  mlir::stablehlo::registerPasses();
  mlir::stablehlo::registerStablehloLegalizeToLinalgPass();
  mlir::registerConvertAffineToStandard();
  mlir::LLVM::registerLLVMPasses();
  mlir::registerLinalgPasses();
  mlir::registerTransformsPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerReconcileUnrealizedCasts();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerSCFToControlFlow();
}

void applyInterpreter(::mlir::OpPassManager &pm, const char *entry_point) {
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = entry_point;
  pm.addPass(mlir::transform::createInterpreterPass(options));
}
}  // namespace

void buildTransformPipeline(::mlir::OpPassManager &pm,
                            const TransformPipelineOptions &options) {
  registerPasses();
  pm.addPass(mlir::llh::createOperationlegalizationPass());
  mlir::transform::PreloadLibraryPassOptions preload_options;
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_LINALG_INCLUDE__);
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_MHLO_INCLUDE__);
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_TENSOR_INCLUDE__);
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_LLVM_INCLUDE__);
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_MEMREF_INCLUDE__);
  pm.addPass(mlir::transform::createPreloadLibraryPass(preload_options));
  pm.addPass(mlir::llh::createRemoveRedundantOpsPass());
  pm.addPass(::mlir::createInlinerPass());
  pm.addPass(mlir::llh::createMarkAotPass());
  pm.addPass(mlir::llh::createInferSymbolShapePass(
      {.CleanSymbolCache = false, .UseEncoding = true}));
  pm.addPass(mlir::llh::createDecomposeOpsPass());
  pm.addPass(mlir::llh::createLoadWeightPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::llh::createTransformLayoutPass(options.targetLayout));
  pm.addPass(mlir::createLLHPreprocessingForHLOPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::llh::createUnloadAndBindEncoding());
  pm.addPass(mlir::llh::createRemoveSymbolPass());
  //===----------------------------------------------------------------------===//
  //  lowing llh
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLLHToArithPass());
  pm.addPass(mlir::createConvertLLHToTensorPass());
  pm.addPass(mlir::createConvertLLHToHLOPass());
  pm.addPass(mlir::index::ex::createFoldIndexCastPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  //  opt mhlo
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_HLO_BASIC_OPT__);
  //===----------------------------------------------------------------------===//
  //  lowing mhlo
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_HLO_TO_LINALG__);
  //===----------------------------------------------------------------------===//
  //  lowing shape
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createShapeToShapeLowering());
  //===----------------------------------------------------------------------===//
  //  tensor lowing
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_TENSOR_BASIC_OPT__);
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  pm.addPass(mlir::llh::createInferSymbolShapePass(
      {.CleanSymbolCache = false, .UseEncoding = false}));
  pm.addPass(mlir::llh::createSymbolCSEPass());
  pm.addPass(mlir::llh::createRemoveSymbolPass());
  //===----------------------------------------------------------------------===//
  //  linalg opt
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_LINALG_BASIC_ANALYSIS__);
  applyInterpreter(pm, __LLC_TRANSFORM_LINALG_BASIC_FUSE__);
  applyInterpreter(pm, __LLC_TRANSFORM_LINALG_BASIC_VECTORIZATION__);
  //===----------------------------------------------------------------------===//
  // bufferization
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_LINALG_BASIC_BUFFERIZATION__);

  //===----------------------------------------------------------------------===//
  // lowing linalg
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  //===----------------------------------------------------------------------===//
  // affine opt
  //===----------------------------------------------------------------------===//
  //   pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  //     mlir::affine::AffineVectorizeOptions vectorize_options;
  //     vectorize_options.vectorSizes = {8};
  //     vectorize_options.vectorizeReductions = true;
  //     pm.addNestedPass<mlir::func::FuncOp>(
  //         mlir::affine::createAffineVectorize(vectorize_options));
  //===----------------------------------------------------------------------===//
  // lowing to csf
  //===----------------------------------------------------------------------===//
  // lowing affine to scf
  pm.addPass(mlir::affine::createLoopFusionPass());
  mlir::VectorTransferToSCFOptions vec_scf_options;
  vec_scf_options.unroll = true;
  vec_scf_options.lowerScalable = true;
  pm.addPass(mlir::createConvertVectorToSCFPass(vec_scf_options));
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createCSEPass());

  //===----------------------------------------------------------------------===//
  // arith opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::arith::createArithExpandOpsPass());

  //===----------------------------------------------------------------------===//
  //  scf  opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  //===----------------------------------------------------------------------===//
  //  gpu
  //===----------------------------------------------------------------------===//
  // pm.addPass(mlir::createForallToParallelLoopPass());
  // pm.addPass(mlir::createParallelLoopFusionPass());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopToGpuPass());
  //===----------------------------------------------------------------------===//
  // memref opt
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_MEMREF_BASIC_OPT__);
  //===----------------------------------------------------------------------===//
  //  lowing to llvm
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_LLVM_LOWING__);
  //===----------------------------------------------------------------------===//
  // LLVM opt
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_LLVM_BASIC_OPT__);
}

void registerTransformPipeline() {
  ::mlir::PassPipelineRegistration<TransformPipelineOptions>(
      "transform-pipeline", "transform pipeline", buildTransformPipeline);
}

}  // namespace llc::pipeline
