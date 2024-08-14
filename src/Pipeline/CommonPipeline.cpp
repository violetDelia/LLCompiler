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
#include "llcompiler/Pipeline/CommonPipeline.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/pass/PassManager.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"

namespace llc::pipleline {

struct CommonPipelineOptions
    : public mlir::PassPipelineOptions<CommonPipelineOptions> {};

void buildCommonPipeline(::mlir::OpPassManager &pm,
                         const CommonPipelineOptions &options) {
  pm.addPass(::mlir::createInlinerPass());       // 内联
  pm.addPass(::mlir::createConvertLLHToTosa());  // LLH lowing to tosa
  pm.addPass(mlir::createSCCPPass());            // 稀疏常量条件传播
  pm.addPass(mlir::tosa::createTosaInferShapesPass());  // 形状推导
  pm.addPass(mlir::createCanonicalizerPass());          // 规范化

  // general pass
  {
    pm.addPass(mlir::createCompositeFixedPointPass());
    pm.addPass(mlir::createSROA());
    pm.addPass(mlir::createTopologicalSortPass());
    pm.addPass(mlir::createPrintOpGraphPass());

    pm.addPass(mlir::createMem2Reg());
  }
  pm.addPass(
      mlir::createControlFlowSinkPass());  // 将算子下沉到使用它的控制块中
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
