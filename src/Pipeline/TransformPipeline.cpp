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
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Pipeline/TransFromPipeline.h"
#include "llcompiler/Support/Enums.h"
#include "llcompiler/TransformLibrary/LibraryConfig.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
namespace llc::pipleline {
void buildTransformPipeline(::mlir::OpPassManager &pm,
                        const TransformPipelineOptions &options) {
  mlir::transform::PreloadLibraryPassOptions transform_otions;
  if (options.target == CPU) {
    transform_otions.transformLibraryPaths.push_back(
        __LLC_TransformLibrary_CPU__);
    pm.addPass(mlir::transform::createPreloadLibraryPass());
  }
  pm.addPass(mlir::transform::createInterpreterPass());
}

void registerTransformPipeline() {
  ::mlir::PassPipelineRegistration<TransformPipelineOptions>(
      "transform-pipeline", "transform pipeline", buildTransformPipeline);
}

}  // namespace llc::pipleline
