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
#include "llcompiler/Pipeline/BasicPipeline.h"

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
#include "llcompiler/Support/Enums.h"
#include "llcompiler/Support/Logger.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"
namespace llc::pipleline {

void buildBasicPipeline(::mlir::OpPassManager &pm,
                        const BasicPipelineOptions &options) {
  mlir::llh::SymbolAnalysis::symbol_enable = options.symbolInfer;
  // 合法化非法的Op
  pm.addPass(mlir::llh::createOperationlegalizationPass());
  //标记Aot算子
  pm.addPass(mlir::llh::createMarkAotPass());
  // 去除冗余Op
  pm.addPass(mlir::llh::createRemoveRedundantOpsPass());
  // 内联
  pm.addPass(::mlir::createInlinerPass());
  // 符号推导和shapeinfer
  pm.addPass(
      mlir::llh::createInferSymbolShapePass({.CleanSymbolCache = false}));
  // 将WeightOp转换为constant
  pm.addPass(mlir::llh::createLoadWeightPass());
  // 布局转换
  pm.addPass(mlir::llh::createTransformLayoutPass());
  // 预处理，这样lowing会方方便一点
  pm.addPass(mlir::createLLHPreprocessingForHLOPass());
  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  // 卸载encodingAttr并绑定到encodingbind Op 上
  pm.addPass(mlir::llh::createUnloadAndBindEncoding());
  pm.addPass(mlir::llh::createRemoveSymbolPass());

  //===----------------------------------------------------------------------===//
  //  lowing llh
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLLHToArithPass());
  pm.addPass(mlir::createConvertLLHToTensorPass());
  pm.addPass(mlir::createConvertLLHToHLOPass());
  pm.addPass(mlir::createConvertLLHToTosaPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::index_ex::createFoldIndexCastPass());

  //===----------------------------------------------------------------------===//
  //  opt mhlo
  //===----------------------------------------------------------------------===//
  //数据流lowing
  //-mhlo-legalize-control-flow
  //-mhlo-sink-constants-to-control-flow
  //去除mhlo.map
  //--mhlo-collapse-elementwise-map
  //简化算子
  //-mhlo-expand-ops-simplifier
  //简化算子 reduce
  // group-reduction-dimensions
  // 规范braodcast-->broadcast-in-dim
  //--hlo-canonicalize-dot
  //--hlo-canonicalize-reduction
  // hlo-legalize-broadcast-to-broadcast-in-dim
  //算子替换
  //--mhlo-legalize-dot-general-to-dot 
  //算子拆解
  //--unfuse_batch_norm
  
  
  //===----------------------------------------------------------------------===//
  //   hlo opt
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloAggressiveFolderPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloAggressiveSimplificationPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloCompatibilityExpanderPass());  //分解
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloLegalizeDeprecatedOpsPass());
  pm.addPass(mlir::stablehlo::createStablehloConvertToSignlessPass());
  pm.addPass(mlir::createCanonicalizerPass());

  //===----------------------------------------------------------------------===//
  //  lowing hlo
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createStablehloQuantLegalizeToTosaRescalePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createStablehloPrepareForTosaPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createStablehloLegalizeToTosaPass());

  //===----------------------------------------------------------------------===//
  //  tosa opt
  //===----------------------------------------------------------------------===//
  // tosa 常量折叠
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaLayerwiseConstantFoldPass(
          {.aggressiveReduceConstant = true}));
  // 算子分解
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaOptionalDecompositions());
  // transpose 消除
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaReduceTransposes());
  pm.addPass(mlir::createCanonicalizerPass());

  //===----------------------------------------------------------------------===//
  //  lowing tosa
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass(
      {.enablePrimitiveOps = true}));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed(
      {.preferConv2DKernelLayoutHWCF = false}));

  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  pm.addPass(mlir::tosa::createTosaToArith(true));

  //===----------------------------------------------------------------------===//
  //  tensor opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::tensor::createFoldTensorSubsetOpsPass());
  pm.addPass(mlir::createCSEPass());

  //===----------------------------------------------------------------------===//
  //  lowing tensor
  //===----------------------------------------------------------------------===//
  // pm.addPass(mlir::transform::createPreloadLibraryPass());
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  //===----------------------------------------------------------------------===//
  //  linalg opt
  //===----------------------------------------------------------------------===//
  //===----------------------------------------------------------------------===//
  //  linalg fusion
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());

  //===----------------------------------------------------------------------===//
  // bufferization
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  mlir::bufferization::OneShotBufferizationOptions bufferization_opts;
  bufferization_opts.bufferizeFunctionBoundaries = true;
  bufferization_opts.allowReturnAllocsFromLoops = true;
  bufferization_opts.copyBeforeWrite = true;
  bufferization_opts.analysisHeuristic = mlir::bufferization::
      OneShotBufferizationOptions::AnalysisHeuristic::BottomUpFromTerminators;
  bufferization_opts.opFilter.allowDialect<
      mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect,
      mlir::linalg::LinalgDialect>();
  // func Bufferize  for remove the memref.copy before one-shot,
  pm.addPass(mlir::func::createFuncBufferizePass());
  // Bufferize
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferization_opts));
  pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());

  mlir::bufferization::BufferResultsToOutParamsOpts buffer_result_opts;
  buffer_result_opts.hoistStaticAllocs = true;
  buffer_result_opts.addResultAttribute = true;
  // Bufferize规范化
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  // 内存inpalce
  pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass(
      buffer_result_opts));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());

  //===----------------------------------------------------------------------===//
  // lowing linalg
  //===----------------------------------------------------------------------===//
  // convert linalg to affine
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createCSEPass());
  // 化简memref.dim
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // 化简memref.dim
  pm.addPass(mlir::memref::createResolveRankedShapeTypeResultDimsPass());

  //===----------------------------------------------------------------------===//
  // affine opt
  //===----------------------------------------------------------------------===//
  // 简化affine表达
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createSimplifyAffineStructuresPass());
  //   //去除affine.delinearize_index
  //   pm.addPass(mlir::affine::createAffineExpandIndexOpsPass());

  //   // affine 并行化
  //   pm.addNestedPass<mlir::func::FuncOp>(
  //       mlir::affine::createAffineParallelizePass());
  // 简化loop
  //   pm.addNestedPass<mlir::func::FuncOp>(
  //       mlir::affine::createLoopCoalescingPass());
  // dma生成
  //   pm.addNestedPass<mlir::func::FuncOp>(
  //       mlir::affine::createAffineDataCopyGenerationPass(0, 3, 0, 1024,
  //                                                        options.L3CacheSize));
  // 简化dma
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createPipelineDataTransferPass());
  // loop融合
  pm.addPass(mlir::affine::createLoopFusionPass(
      0, 1024, false, mlir::affine::FusionMode::Greedy));
  // 去除冗余内存操作
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createAffineScalarReplacementPass());
  // 循环展开+融合
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopUnrollAndJamPass());
  // 循环合并
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopCoalescingPass());
  //   pm.addNestedPass<mlir::func::FuncOp>(
  //       mlir::affine::createAffineDataCopyGenerationPass(3, 2, 0, 1024,
  //                                                        options.L2CacheSize));
  //   pm.addNestedPass<mlir::func::FuncOp>(
  //       mlir::affine::createAffineDataCopyGenerationPass(2, 1, 0, 128,
  //                                                        options.L1CacheSize));

  //   pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopTilingPass(128));

  //===----------------------------------------------------------------------===//
  // lowing affine
  //===----------------------------------------------------------------------===//
  // lowing affine to scf
  pm.addPass(mlir::createLowerAffinePass());

  //===----------------------------------------------------------------------===//
  // arith opt
  //===----------------------------------------------------------------------===//
  // 消除不支持的浮点数
  pm.addPass(mlir::arith::createArithEmulateUnsupportedFloats());
  // 转换为Unsigned
  pm.addPass(mlir::arith::createArithUnsignedWhenEquivalentPass());
  // 优化int的位宽
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  // 合法化lowing to LLVM
  pm.addPass(mlir::arith::createArithExpandOpsPass());

  //===----------------------------------------------------------------------===//
  //  scf  opt
  //===----------------------------------------------------------------------===//
  //   // forall ->for.para
  //   pm.addPass(mlir::createForallToParallelLoopPass());
  // 固定变量移到loop外
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  // 固定Op移到loop外
  pm.addPass(mlir::createLoopInvariantSubsetHoistingPass());
  // 剥离循环首尾
  pm.addPass(mlir::createForLoopPeelingPass());
  // 外移计算
  pm.addPass(mlir::createForLoopRangeFoldingPass());
  // 控制流下沉
  pm.addPass(mlir::createControlFlowSinkPass());
  // scf fusion
  pm.addPass(mlir::createParallelLoopFusionPass());
  // 规范化scf
  pm.addPass(mlir::createSCFForLoopCanonicalizationPass());
  // 条件常量传播
  pm.addPass(mlir::createSCCPPass());
  //   // for -> while
  //   pm.addPass(mlir::createForToWhileLoopPass());
  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  // memref opt
  //===----------------------------------------------------------------------===//
  // 化简memref.dim
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // 化简memref.dim
  pm.addPass(mlir::memref::createResolveRankedShapeTypeResultDimsPass());
  // 内存访问转为reinterpret_cast
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  // 控制块alloc化简
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());
  // loop中控制块alloc化简
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());
  // 优化realloc为scf->alloc
  pm.addPass(mlir::memref::createExpandReallocPass());
  // 内存折叠
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  // 规范化
  pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  // 添加内存释放op
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferDeallocationPass());
  // liveness
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createOptimizeAllocationLivenessPass());

  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  // legalizes memref dialect ops to be convertible to LLVM
  pm.addPass(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::createBufferizationToMemRefPass());
  // 去重重复的func
  pm.addPass(mlir::func::createDuplicateFunctionEliminationPass());
  //===----------------------------------------------------------------------===//
  // lowing scf
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertSCFToCFPass());

  //===----------------------------------------------------------------------===//
  // lowing to llvm
  //===----------------------------------------------------------------------===//
  if (options.target == TARGET::CPU) {
    pm.addPass(mlir::createConvertFuncToLLVMPass(
        {.useBarePtrCallConv = false, .indexBitwidth = options.indexBitWidth}));
    pm.addPass(mlir::createConvertControlFlowToLLVMPass(
        {.indexBitwidth = options.indexBitWidth}));
    pm.addPass(mlir::createConvertIndexToLLVMPass(
        {.indexBitwidth = options.indexBitWidth}));
    pm.addPass(mlir::createArithToLLVMConversionPass(
        {.indexBitwidth = options.indexBitWidth}));
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass(
        {.useAlignedAlloc = false,
         .indexBitwidth = options.indexBitWidth,
         .useGenericFunctions = false}));

    // lowing to scf to cf
  } else {
    FATAL(llc::MLIR);
  }
  //===----------------------------------------------------------------------===//
  // LLVM opt
  //===----------------------------------------------------------------------===//
  // 适配执行引擎
  pm.addPass(mlir::LLVM::ex::createAdaptEntryParmsForEnginePass());
  // 合法化
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
  // 内存转寄存器
  pm.addPass(mlir::createMem2Reg());
  // 公共表达式消除
  pm.addPass(mlir::createCSEPass());
  // 死符号消除
  pm.addPass(mlir::createSymbolDCEPass());
  // 规范化
  pm.addPass(mlir::createCanonicalizerPass());
  // 消除多余的未定义转换
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}
void registerBasicPipeline() {
  ::mlir::PassPipelineRegistration<BasicPipelineOptions>(
      "basic-pipeline", "basic pipeline", buildBasicPipeline);
}

}  // namespace llc::pipleline
