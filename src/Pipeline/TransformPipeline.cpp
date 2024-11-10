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
#include "llcompiler/Conversion/LLHToArith/LLHToArith.h"
#include "llcompiler/Conversion/LLHToHLO/LLHToHLO.h"
#include "llcompiler/Conversion/LLHToTensor/LLHToTensor.h"
#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
#include "llcompiler/Conversion/Passes.h"
#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/TosaExtension/Transforms/Passes.h"
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/TransformLibrary/LibraryEntry.h"
#include "llcompiler/TransformLibrary/LibraryPath.h"
#include "mhlo/IR/register.h"
#include "mhlo/interfaces/bufferizable_op_interface_impl.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "transforms/passes.h"
namespace llc::pipeline {

namespace {
void registerPasses() {
  mlir::llh::registerLLHOptPasses();
  mlir::registerLLCConversionPasses();
  mlir::index::ex::registerIndexExtensionPasses();
  mlir::LLVM::ex::registerLLVMExtensionPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::hlo::registerFinalBufferizePass();
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
  pm.addPass(mlir::transform::createPreloadLibraryPass(preload_options));
  // 合法化非法的Op
  pm.addPass(mlir::llh::createOperationlegalizationPass());
  // 标记Aot算子
  pm.addPass(mlir::llh::createMarkAotPass());
  // 去除冗余Op
  pm.addPass(mlir::llh::createRemoveRedundantOpsPass());
  // 内联
  pm.addPass(::mlir::createInlinerPass());
  // 符号推导和shapeinfer
  pm.addPass(mlir::llh::createInferSymbolShapePass(
      {.CleanSymbolCache = false, .UseBinding = true}));
  // 将WeightOp转换为constant
  pm.addPass(mlir::llh::createLoadWeightPass());
  // 布局转换
  pm.addPass(mlir::llh::createTransformLayoutPass(options.targetLayout));
  // 预处理，这样lowing会方方便一点
  pm.addPass(mlir::createLLHPreprocessingForHLOPass());
  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  // 卸载encodingAttr并绑定到encodingbind Op 上
  pm.addPass(mlir::llh::createUnloadAndBindEncoding());
  // 去除符号，因为hlo的规范化不识别
  pm.addPass(mlir::llh::createRemoveSymbolPass());
  //===----------------------------------------------------------------------===//
  //  lowing llh
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLLHToArithPass());
  pm.addPass(mlir::createConvertLLHToTensorPass());
  pm.addPass(mlir::createConvertLLHToHLOPass());
  pm.addPass(mlir::index::ex::createFoldIndexCastPass());
  //===----------------------------------------------------------------------===//
  //  opt mhlo
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_MLHO_BASIC_OPT__);
  //===----------------------------------------------------------------------===//
  //  lowing mhlo
  //===----------------------------------------------------------------------===//
    pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeToStdPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createSymbolicShapeOptimizationPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createLegalizeHloToLinalgPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::mhlo::createLegalizeControlFlowPass());
  // NOTE: unkown error (mutithreading)
  applyInterpreter(pm, __LLC_TRANSFORM_MLHO_TO_LINALG__);
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
  //===----------------------------------------------------------------------===//
  //  linalg opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  // applyInterpreter(pm, __LLC_TRANSFORM_LINALG_SPECIALIZE__);
  applyInterpreter(pm, __LLC_TRANSFORM_LINALG_GENERALIZE__);
  pm.addPass(mlir::createLinalgInlineScalarOperandsPass());
  pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgDetensorizePass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  // bufferization
  //===----------------------------------------------------------------------===//
  applyInterpreter(pm, __LLC_TRANSFORM_MLHO_BUFFERIZE__);

  //===----------------------------------------------------------------------===//
  // lowing linalg
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::hlo::createAllocToArgPass());
}

void registerTransformPipeline() {
  ::mlir::PassPipelineRegistration<TransformPipelineOptions>(
      "transform-pipeline", "transform pipeline", buildTransformPipeline);
}

}  // namespace llc::pipeline