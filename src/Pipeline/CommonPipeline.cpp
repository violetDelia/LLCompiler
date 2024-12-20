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

#include "llcompiler/Pipeline/CommonPipeline.h"

#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h"
#include "llcompiler/Dialect/TosaExtension/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Affine/Passes.h"
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
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace llc::pipeline {

void buildCommonPipeline(::mlir::OpPassManager &pm,
                         const CommonPipelineOptions &options) {
  //===----------------------------------------------------------------------===//
  // options
  //===----------------------------------------------------------------------===//
  mlir::tosa::TosaValidationOptions ValidationOption;
  // if (options.runMode == MODE::Inference) {
  //   ValidationOption.profile = mlir::tosa::TosaProfileEnum::MainInference;
  // }
  // if (options.runMode == MODE::Training) {
  //   ValidationOption.profile = mlir::tosa::TosaProfileEnum::MainTraining;
  // }
  mlir::TosaToLinalgOptions TosaToLinalgOption{
      .disableTosaDecompositions = true, .aggressiveReduceConstant = false};
  if (options.ReduceConstant) {
    TosaToLinalgOption.aggressiveReduceConstant = true;
  }

  //===----------------------------------------------------------------------===//
  // llh
  //===----------------------------------------------------------------------===//
  // 内联
  pm.addPass(::mlir::createInlinerPass());
  // LLH lowing to tosa
  pm.addPass(::mlir::createConvertLLHToTosaPass());
  //===----------------------------------------------------------------------===//
  // tosa opt
  //===----------------------------------------------------------------------===//

  // 布局转换到NHWC
  // 布局转换到NHWC
  pm.addPass(mlir::tosa_ex::createTransformLayoutToNHWCPass());
  if (!options.onlyCompiler && options.optInTosa) {
    // 算子分解
    pm.addPass(mlir::tosa::createTosaOptionalDecompositions());
    // 规范化
    pm.addPass(mlir::createCanonicalizerPass());
    // 形状推导
    pm.addPass(mlir::tosa::createTosaInferShapesPass());
  }
  // 规范算子广播形状
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  // 检测算子合法性
  pm.addPass(mlir::tosa::createTosaValidation(ValidationOption));
  if (!options.onlyCompiler && options.optInTosa) {
    // cse
    pm.addPass(mlir::createCSEPass());
    // 死代码消除
    pm.addPass(mlir::createRemoveDeadValuesPass());
    // 规范化
    pm.addPass(mlir::createCanonicalizerPass());
  }
  //===----------------------------------------------------------------------===//
  // lowing tosa
  //===----------------------------------------------------------------------===//
  // Tosa lowing to linalg step1
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed(
      {.preferConv2DKernelLayoutHWCF = false}));
  // 规范算子广播形状
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  if (!options.onlyCompiler && options.optInTosa) {
    // 常量折叠
    pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass(
        {.aggressiveReduceConstant = options.ReduceConstant}));
    // cse
    pm.addPass(mlir::createCSEPass());
    // 规范化
    pm.addPass(mlir::createCanonicalizerPass());
  }
  // Tosa lowing to linalg step2
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  // Tosa lowing to tensor
  pm.addPass(mlir::tosa::createTosaToTensor());
  // Tosa lowing to arith
  pm.addPass(mlir::tosa::createTosaToArith(true));
  // Tosa lowing to scf
  pm.addPass(mlir::tosa::createTosaToSCF());
  //===----------------------------------------------------------------------===//
  // lowing tensor and tensor opt
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.optInTensor) {
    // tensor.insert_slice  的常量折叠
    pm.addPass(mlir::tensor::createFoldTensorSubsetOpsPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }
  // tensor.pad -> linalg
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  //===----------------------------------------------------------------------===//
  //  linalg fusion
  //===----------------------------------------------------------------------===//
  // named linalg lowing to  linalg.generic
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  if (!options.onlyCompiler && options.optInLinalg) {
    // 单op转换为linalg.generic
    pm.addPass(mlir::createConvertElementwiseToLinalgPass());
    // linalg.generic fusion
    pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
    // 规范化
    pm.addPass(mlir::createCanonicalizerPass());
  }

  //===----------------------------------------------------------------------===//
  // bufferization
  //===----------------------------------------------------------------------===//
  mlir::bufferization::OneShotBufferizationOptions bufferization_opts;
  bufferization_opts.bufferizeFunctionBoundaries = true;
  bufferization_opts.analysisHeuristic = mlir::bufferization::
      OneShotBufferizationOptions::AnalysisHeuristic::BottomUpFromTerminators;
  bufferization_opts.opFilter.allowDialect<
      mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect,
      mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
      mlir::func::FuncDialect>();
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(
      bufferization_opts));                           // Bufferize
  pm.addPass(mlir::func::createFuncBufferizePass());  // func Bufferize
  mlir::bufferization::BufferResultsToOutParamsOpts buffer_result_opts;
  buffer_result_opts.filterFn = [](mlir::func::FuncOp *func) {
    auto name = func->getSymName();
    return (name != "main");
  };
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass(
      buffer_result_opts));                     // 内存inpalce
  pm.addPass(mlir::createCanonicalizerPass());  // 规范化

  //===----------------------------------------------------------------------===//
  // lowing linalg to affine
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());  // convert linalg
                                                             // to affine
  //===----------------------------------------------------------------------===//
  // lowing affine to scf and affine opt
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.usingAffine) {
    pm.addPass(
        mlir::affine::
            createAffineExpandIndexOpsPass());  // 去除affine.delinearize_index
    // pm.addPass(mlir::affine::createAffineDataCopyGenerationPass(3, 1, 0,
    // 1024,1024 *1024));//添加dma
    pm.addPass(mlir::affine::createLoopFusionPass(
        0, 1024 * 1024, false,
        mlir::affine::FusionMode::Greedy));                    // loop融合
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());  // 内存折叠
    pm.addPass(
        mlir::affine::createAffineScalarReplacementPass());  // 去除冗余内存操作
    pm.addPass(mlir::affine::createPipelineDataTransferPass());  // 简化dma
  }
  // mlir::affine::AffineVectorizeOptions vector_options;
  // vector_options.vectorSizes = {1024};
  // pm.addPass(mlir::affine::createAffineVectorize(vector_options));  //
  // 向量化
  if (!options.onlyCompiler && options.usingAffine) {
    pm.addPass(mlir::affine::createLoopUnrollAndJamPass());  // 循环展开+融合
    pm.addPass(mlir::affine::createLoopCoalescingPass());     // 循环合并
    pm.addPass(mlir::affine::createAffineParallelizePass());  // affine 并行化
    pm.addPass(mlir::affine::createLoopCoalescingPass());     // 简化loop
    pm.addPass(
        mlir::affine::createSimplifyAffineStructuresPass());  // 简化affine表达
  }
  pm.addPass(mlir::createLowerAffinePass());  // lowing affine to scf
  //===----------------------------------------------------------------------===//
  // arith opt
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.optInArith) {
    pm.addPass(
        mlir::arith::
            createArithEmulateUnsupportedFloats());  // 消除不支持的浮点数
    pm.addPass(mlir::arith::
                   createArithUnsignedWhenEquivalentPass());  // 转换为Unsigned
    pm.addPass(
        mlir::arith::createIntRangeOptimizationsPass());  // 优化int的位宽
    pm.addPass(mlir::createCSEPass());                    // cse
    pm.addPass(mlir::createCanonicalizerPass());          // 规范化
  }
  pm.addPass(mlir::arith::createArithExpandOpsPass());  // 合法化lowing to LLVM
  //===----------------------------------------------------------------------===//
  // memref opt step
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.optInMemref) {
    pm.addPass(
        mlir::bufferization::createBufferHoistingPass());  // 控制块alloc化简
    pm.addPass(mlir::bufferization::
                   createBufferLoopHoistingPass());  // loop中控制块alloc化简
    pm.addPass(
        mlir::memref::createExpandReallocPass());  // 优化realloc为scf->alloc
    pm.addPass(mlir::memref::
                   createResolveShapedTypeResultDimsPass());  // 化简memref.dim
    pm.addPass(
        mlir::memref::
            createResolveRankedShapeTypeResultDimsPass());  // 化简memref.dim
    pm.addPass(
        mlir::memref::
            createExpandStridedMetadataPass());  // 内存访问转为reinterpret_cast
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());  // 规范化
    pm.addPass(mlir::createCSEPass());                       // cse
    pm.addPass(mlir::createCanonicalizerPass());             // 规范化
  }  // Normalize  memrefs
  pm.addPass(
      mlir::bufferization::createBufferDeallocationPass());  // 添加内存释放op
  pm.addPass(
      mlir::memref::createExpandOpsPass());  // legalizes memref dialect ops
  // to be convertible to LLVM
  //===----------------------------------------------------------------------===//
  //  scf  opt
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.optInSCF) {
    pm.addPass(mlir::createForallToParallelLoopPass());  // forall ->for.para
    pm.addPass(
        mlir::createLoopInvariantCodeMotionPass());  // 固定变量移到loop外
    pm.addPass(
        mlir::createLoopInvariantSubsetHoistingPass());  // 固定Op移到loop外
    pm.addPass(mlir::createForLoopPeelingPass());        // 剥离循环首尾
    pm.addPass(mlir::createForLoopRangeFoldingPass());   // 外移计算
    pm.addPass(mlir::createControlFlowSinkPass());       // 控制流下沉
    pm.addPass(mlir::createParallelLoopFusionPass());    // scf fusion
    pm.addPass(mlir::createSCFForLoopCanonicalizationPass());  // 规范化scf
    pm.addPass(mlir::createSCCPPass());            // 条件常量传播
    pm.addPass(mlir::createForToWhileLoopPass());  // for -> while
    pm.addPass(mlir::createCanonicalizerPass());   // 规范化
  }
  //===----------------------------------------------------------------------===//
  // lowing to scf to cf
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertSCFToCFPass());
  //===----------------------------------------------------------------------===//
  // lowing to last ir
  //===----------------------------------------------------------------------===//
  pm.addPass(
      mlir::func::createDuplicateFunctionEliminationPass());  // 去重重复的func
  pm.addPass(mlir::createConvertControlFlowToSPIRVPass());
  if (options.target == TARGET::CPU) {
    pm.addPass(mlir::createConvertFuncToLLVMPass(
        {.useBarePtrCallConv = false, .indexBitwidth = options.indexBitWidth}));
    pm.addPass(mlir::createConvertControlFlowToLLVMPass(
        {.indexBitwidth = options.indexBitWidth}));
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(
        {.useAlignedAlloc = false,
         .indexBitwidth = options.indexBitWidth,
         .useGenericFunctions = false}));
    pm.addPass(mlir::createArithToLLVMConversionPass(
        {.indexBitwidth = options.indexBitWidth}));
  }
  //===----------------------------------------------------------------------===//
  // opt last ir
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.optInLLVM) {
    pm.addPass(mlir::LLVM::createLegalizeForExportPass());  // 合法化
  }
  pm.addPass(mlir::createMem2Reg());            // 内存转寄存器
  pm.addPass(mlir::createCSEPass());            // 公共表达式消除
  pm.addPass(mlir::createSymbolDCEPass());      // 死符号消除
  pm.addPass(mlir::createCanonicalizerPass());  // 规范化
  pm.addPass(
      mlir::createReconcileUnrealizedCastsPass());  // 消除多余的未定义转换
}

void registerCommonPipeline() {
  ::mlir::PassPipelineRegistration<CommonPipelineOptions>(
      "common-pipeline", "The default pipeline for LLC", buildCommonPipeline);
}
// namespace llc
}  // namespace llc::pipeline
