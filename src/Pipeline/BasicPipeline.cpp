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
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Pipeline/BasicPipeline.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
namespace llc::pipleline {

void buildBasicPipeline(::mlir::OpPassManager &pm,
                        const BasicPipelineOptions &options) {
  pm.addPass(mlir::llh::createOperationlegalizationPass());  //合法化非法的Op
  pm.addPass(mlir::llh::createRemoveRedundantOpsPass());     // 去除冗余Op
  pm.addPass(::mlir::createInlinerPass());                   // 内联
  pm.addPass(mlir::llh::createReshapeBeforeBraodcastPass());  // 广播前插入reshape
  pm.addPass(mlir::llh::createInferSymbolShapePass());  // 符号推导和shapeinfer
  pm.addPass(mlir::createCanonicalizerPass());          //规范化
  pm.addPass(mlir::llh::createLoadWeightPass());  //将WeightOp转换为constant
  pm.addPass(mlir::createCanonicalizerPass());    //规范化
}
void registerBasicPipeline() {
  ::mlir::PassPipelineRegistration<BasicPipelineOptions>(
      "basic-pipeline", "basic pipeline", buildBasicPipeline);
}

}  // namespace llc::pipleline
