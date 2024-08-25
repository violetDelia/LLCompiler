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
#include "llcompiler/Dialect/TosaExtension/Transforms/Passes.h"
#include "llcompiler/Pipeline/Enums.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/pass/PassManager.h"

namespace llc::pipleline {

struct CommonPipelineOptions
    : public mlir::PassPipelineOptions<CommonPipelineOptions> {
  Option<bool> printOpGraph{*this, "print-op-graph",
                            llvm::cl::desc("use PrintOpGraphPass."),
                            llvm::cl::init(false)};
  Option<bool> ReduceConstant{*this, "reduce-const",
                              llvm::cl::desc("Reduce constant aggressive"),
                              llvm::cl::init(true)};
  Option<RUN_MODE> runMode{
      *this, "mode", llvm::cl::desc("run mode"),
      llvm::cl::init(RUN_MODE::INFERENCE),
      llvm::cl::values(clEnumValN(RUN_MODE::INFERENCE,
                                  run_mode_to_str(RUN_MODE::INFERENCE), ""),
                       clEnumValN(RUN_MODE::TRAINING,
                                  run_mode_to_str(RUN_MODE::TRAINING), ""))};
  Option<bool> onlyCompiler{*this, "only-compiler",
                            llvm::cl::desc("only compiler ther model"),
                            llvm::cl::init(false)};
  Option<bool> optInTosa{*this, "opt-tosa",
                         llvm::cl::desc("optimization in tosa dialcet"),
                         llvm::cl::init(true)};
  Option<bool> optInTensor{*this, "opt-tensor",
                           llvm::cl::desc("optimization in tensor dialcet"),
                           llvm::cl::init(true)};
  Option<bool> optInLinalg{*this, "opt-linalg",
                           llvm::cl::desc("optimization in linalg dialcet"),
                           llvm::cl::init(true)};
  Option<bool> optInMemref{*this, "opt-memref",
                           llvm::cl::desc("optimization in memref dialcet"),
                           llvm::cl::init(true)};
  Option<bool> usingAffine{*this, "use-affine",
                           llvm::cl::desc("optimization in affine dialcet"),
                           llvm::cl::init(true)};
  Option<bool> optInSCF{*this, "opt-scf",
                        llvm::cl::desc("optimization in scf dialcet"),
                        llvm::cl::init(true)};
};

void buildCommonPipeline(::mlir::OpPassManager &pm,
                         const CommonPipelineOptions &options) {
  //===----------------------------------------------------------------------===//
  // options
  //===----------------------------------------------------------------------===//
  mlir::tosa::TosaValidationOptions ValidationOption;
  if (options.runMode == RUN_MODE::INFERENCE) {
    ValidationOption.profile = mlir::tosa::TosaProfileEnum::MainInference;
  }
  if (options.runMode == RUN_MODE::TRAINING) {
    ValidationOption.profile = mlir::tosa::TosaProfileEnum::MainTraining;
  }
  mlir::TosaToLinalgOptions TosaToLinalgOption{
      .disableTosaDecompositions = true, .aggressiveReduceConstant = false};
  if (options.ReduceConstant) {
    TosaToLinalgOption.aggressiveReduceConstant = true;
  }

  //===----------------------------------------------------------------------===//
  // llh
  //===----------------------------------------------------------------------===//
  pm.addPass(::mlir::createInlinerPass());       // 内联
  pm.addPass(::mlir::createConvertLLHToTosa());  // LLH lowing to tosa
  //===----------------------------------------------------------------------===//
  // tosa opt
  //===----------------------------------------------------------------------===//
  pm.addPass(
      mlir::tosa_ex::createTransformLayoutToNHWCPass());  // 布局转换到NHWC
  if (!options.onlyCompiler && options.optInTosa) {
    pm.addPass(mlir::tosa::createTosaOptionalDecompositions());  // 算子分解
    pm.addPass(mlir::createCanonicalizerPass());                 // 规范化
    pm.addPass(mlir::tosa::createTosaInferShapesPass());         // 形状推导
  }
  pm.addPass(
      mlir::tosa::createTosaMakeBroadcastablePass());  // 规范算子广播形状
  pm.addPass(
      mlir::tosa::createTosaValidation(ValidationOption));  // 检测算子合法性
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  // lowing tosa
  //===----------------------------------------------------------------------===//
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed(
      {.preferConv2DKernelLayoutHWCF = false}));  // Tosa lowing to linalg step1
  pm.addPass(
      mlir::tosa::createTosaMakeBroadcastablePass());  // 规范算子广播形状
  if (!options.onlyCompiler && options.optInTosa) {
    pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass(
        {.aggressiveReduceConstant = options.ReduceConstant}));  // 常量折叠
    pm.addPass(mlir::createCanonicalizerPass());                 // 规范化
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tosa::createTosaToLinalg());         // Tosa lowing to linalg step2
  pm.addPass(mlir::tosa::createTosaToTensor());  // Tosa lowing to tensor
  pm.addPass(mlir::tosa::createTosaToArith(true));  // Tosa lowing to arith
  pm.addPass(mlir::tosa::createTosaToSCF());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  // tensor opt
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.optInTensor) {
    pm.addPass(
        mlir::tensor::createFoldTensorSubsetOpsPass());  // tensor.insert_slice
    // 的常量折叠
  }
  //===----------------------------------------------------------------------===//
  // lowing tensor
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertTensorToLinalgPass());  // tensor.pad -> linalg
  //===----------------------------------------------------------------------===//
  //  linalg fusion
  //===----------------------------------------------------------------------===//
  pm.addPass(
      mlir::createLinalgGeneralizeNamedOpsPass());  // named linalg lowing to
                                                    // linalg.generic
  if (!options.onlyCompiler && options.optInLinalg) {
    pm.addPass(
        mlir::
            createConvertElementwiseToLinalgPass());  // 单op转换为linalg.generic
    pm.addPass(
        mlir::createLinalgElementwiseOpFusionPass());  // linalg.generic fusion
  }
  //===----------------------------------------------------------------------===//
  // bufferization
  //===----------------------------------------------------------------------===//
  mlir::bufferization::OneShotBufferizationOptions tensor_buffer_opts;
  tensor_buffer_opts.analysisHeuristic = mlir::bufferization::
      OneShotBufferizationOptions::AnalysisHeuristic::BottomUpFromTerminators;
  tensor_buffer_opts.opFilter.allowDialect<
      mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect,
      mlir::linalg::LinalgDialect, mlir::arith::ArithDialect,
      mlir::func::FuncDialect>();
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(
      tensor_buffer_opts));  // 内存化张量
  // pm.addPass(mlir::bufferization::createFinalizingBufferizePass());
  pm.addPass(mlir::createCanonicalizerPass());  // 规范化
  //===----------------------------------------------------------------------===//
  // lowing linalg to affine
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());  // convert linalg
                                                             // to affine
  //===----------------------------------------------------------------------===//
  // memref opt step1
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
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());  // 内存折叠
    pm.addPass(
        mlir::memref::
            createExpandStridedMetadataPass());  // 内存访问转为reinterpret_cast

    pm.addPass(mlir::memref::createNormalizeMemRefsPass());  // 规范化
  }  // Normalize  memrefs
  pm.addPass(
      mlir::bufferization::createBufferDeallocationPass());  // 添加内存释放op
  pm.addPass(
      mlir::memref::createExpandOpsPass());  // legalizes memref dialect ops to
                                             // be convertible to LLVM
  pm.addPass(mlir::createCSEPass());         // cse
  pm.addPass(mlir::createCanonicalizerPass());  // 规范化
  //===----------------------------------------------------------------------===//
  // affine opt
  //===----------------------------------------------------------------------===//
  if (!options.onlyCompiler && options.usingAffine) {
    pm.addPass(
        mlir::affine::
            createAffineExpandIndexOpsPass());  // 去除affine.delinearize_index
    // pm.addPass(mlir::affine::createAffineDataCopyGenerationPass(3, 1, 0,
    // 1024,1024 *1024));//添加dma
    pm.addPass(mlir::affine::createLoopFusionPass(
        0, 1024 * 1024, false, mlir::affine::FusionMode::Greedy));  // loop融合
    pm.addPass(
        mlir::affine::createAffineScalarReplacementPass());  // 去除冗余内存操作
    pm.addPass(mlir::affine::createPipelineDataTransferPass());  // 简化dma
  }
  pm.addPass(mlir::affine::createAffineVectorize());  // 向量化
  if (!options.onlyCompiler && options.usingAffine) {
    pm.addPass(mlir::affine::createLoopCoalescingPass());  // 简化loop
    pm.addPass(
        mlir::affine::createSimplifyAffineStructuresPass());  // 简化affine表达
  }
  //===----------------------------------------------------------------------===//
  // lowing affine to scf and opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createLowerAffinePass());  // lowing affine to scf
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());  // 固定变量移到loop外
  pm.addPass(
      mlir::createLoopInvariantSubsetHoistingPass());  // 固定Op移到loop外

  pm.addPass(mlir::createCanonicalizerPass());  // 规范化
  if (!options.onlyCompiler && options.optInSCF) {
  }
  //===----------------------------------------------------------------------===//
  // lowing to scf to cf
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createConvertSCFToCFPass());  // lowing to scf to cf

  //===----------------------------------------------------------------------===//
  // lowing to llvm
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createControlFlowSinkPass());  // 控制流下沉
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  //===----------------------------------------------------------------------===//
  //  lowing arith
  //===----------------------------------------------------------------------===//
  //===----------------------------------------------------------------------===//
  //  memref opt step2
  //===----------------------------------------------------------------------===//
  //  normalize for llvm
  //  tensor -> memref
  //  pm.addPass(mlir::tosa::createTosaToArith(true, true));
  //  pm.addPass(mlir::tosa::createTosaToTensor());
  //  pm.addPass(mlir::tosa::createTosaToSCF());
  //  pm.addPass(mlir::createConvertTensorToLinalgPass());
  //  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  //  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  //===----------------------------------------------------------------------===//
  //  lowing tosa
  //===----------------------------------------------------------------------===//
  //  if (options.printOpGraph) {
  //    pm.addPass(mlir::createPrintOpGraphPass());  // 输出Op Graph
  //  }  // 输出Op graph
  //  pm.addPass(mlir::createCSEPass());               // 公共表达式消除
  //  pm.addPass(mlir::createRemoveDeadValuesPass());  // 死代码消除
  //  pm.addPass(mlir::createSymbolDCEPass());         // 死符号消除
}

void registerCommonPipeline() {
  ::mlir::PassPipelineRegistration<CommonPipelineOptions>(
      "common-pipeline", "The default pipeline for LLC", buildCommonPipeline);
}
// namespace llc
}  // namespace llc::pipleline
