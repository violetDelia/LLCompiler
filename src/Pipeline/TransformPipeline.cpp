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
#include "llcompiler/TransformLibrary/LibraryConfig.h"
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
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
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
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
#include "transforms/passes.h"
namespace llc::pipeline {

void applyInterpreter(::mlir::OpPassManager &pm, const char *entry_point) {
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = entry_point;
  pm.addPass(mlir::transform::createInterpreterPass(options));
}

void buildTransformPipeline(::mlir::OpPassManager &pm,
                            const TransformPipelineOptions &options) {
  pm.addPass(mlir::llh::createOperationlegalizationPass());
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
  pm.addPass(mlir::createConvertLLHToTosaPass());
  //===----------------------------------------------------------------------===//
  //  opt mhlo
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createCanonicalizerPass());
  // 简化reshape  braodcast
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSymbolicShapeOptimizationPass());
  // lowing shape 相关的计算
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());
  // 去除tuple
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createFlattenTuplePass());
  // 简化reduce
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createGroupReductionDimensionsPass());
  // 规范braodcast-->broadcast-in-dim
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeBroadcastToBroadcastInDimPass());
  // 规范化 dot
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeDotPass());
  // 规范化reduce
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeReductionPass());
  // 规范化 gather
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeGatherPass());
  // 规范化scatter
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createHloCanonicalizeScatterPass());
  // 移动常量到控制流内部
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSinkConstantsToControlFlowPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // 简化算子
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createMhloExpandOpsSimplifierPass());
  // 简化算子 reduce
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createGroupReductionDimensionsPass());
  // 算子替换
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeDotToDotGeneralPass());
  //算子拆解BatchNorm
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createTestUnfuseBatchNormPass());
  // 传播广播
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  //  lowing mhlo
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeToStdPass());
  //pm.addPass(mlir::mhlo::createLegalizeToMemrefPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeHloToLinalgPass(true));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeControlFlowPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  //  lowing shape
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createShapeToShapeLowering());
  //===----------------------------------------------------------------------===//
  //  tensor lowing
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::tensor::createFoldTensorSubsetOpsPass());
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  //===----------------------------------------------------------------------===//
  //  linalg opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  pm.addPass(mlir::createLinalgSpecializeGenericOpsPass());
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
pm.addPass(mlir::createLinalgInlineScalarOperandsPass());
pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  // bufferization
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::hlo::createOneShotBufferizePass());
  pm.addPass(mlir::createFinalBufferizePass(64));
  pm.addPass(mlir::createCanonicalizerPass());
}

void registerTransformPipeline() {
  ::mlir::PassPipelineRegistration<TransformPipelineOptions>(
      "transform-pipeline", "transform pipeline", buildTransformPipeline);
}

}  // namespace llc::pipeline
