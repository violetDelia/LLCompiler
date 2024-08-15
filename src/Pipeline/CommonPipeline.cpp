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
#include "llcompiler/Pipeline/Enums.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
};

void buildCommonPipelineTosaOpt(::mlir::OpPassManager &pm,
                                const CommonPipelineOptions &options) {
  // 规范化
}

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
  //===----------------------------------------------------------------------===//
  // llh
  //===----------------------------------------------------------------------===//
  pm.addPass(::mlir::createInlinerPass());       // 内联
  pm.addPass(::mlir::createConvertLLHToTosa());  // LLH lowing to tosa
  //===----------------------------------------------------------------------===//
  // tosa opt
  //===----------------------------------------------------------------------===//
  pm.addPass(mlir::tosa::createTosaOptionalDecompositions());  // 算子分解
  pm.addPass(mlir::createCanonicalizerPass());                 // 规范化
  pm.addPass(mlir::tosa::createTosaInferShapesPass());         // 形状推导
  pm.addPass(
      mlir::tosa::createTosaMakeBroadcastablePass());  // 规范算子广播形状
  pm.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass(
      {.aggressiveReduceConstant = options.ReduceConstant}));  // 常量折叠
  pm.addPass(
      mlir::tosa::createTosaValidation(ValidationOption));  // 检测算子合法性
  pm.addPass(mlir::createCanonicalizerPass());
  //===----------------------------------------------------------------------===//
  // lowing tosa
  //===----------------------------------------------------------------------===//

  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalgNamed(
      {.preferConv2DKernelLayoutHWCF = false}));
  // pm.addPass(mlir::tosa::createTosaToArith(true, true));
  // pm.addPass(mlir::tosa::createTosaToTensor());
  // pm.addPass(mlir::tosa::createTosaToSCF());
  // pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  //===----------------------------------------------------------------------===//
  // lowing tosa
  //===----------------------------------------------------------------------===//
  if (options.printOpGraph) {
    pm.addPass(mlir::createPrintOpGraphPass());  // 输出Op Graph
  }  // 输出Op graph
  pm.addPass(mlir::createCSEPass());               // 公共表达式消除
  pm.addPass(mlir::createRemoveDeadValuesPass());  // 死代码消除
  pm.addPass(mlir::createSymbolDCEPass());         // 死符号消除
}

void registerCommonPipeline() {
  ::mlir::PassPipelineRegistration<CommonPipelineOptions>(
      "common-pipeline", "The default pipeline for LLC", buildCommonPipeline);
}
// namespace llc
}  // namespace llc::pipleline
