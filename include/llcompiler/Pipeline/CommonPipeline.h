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

#ifndef INCLUDE_LLCOMPILER_PIPELINE_COMMONPIPELINE_H_
#define INCLUDE_LLCOMPILER_PIPELINE_COMMONPIPELINE_H_
#include "llcompiler/Support/Enums.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace llc::pipleline {
struct CommonPipelineOptions
    : public mlir::PassPipelineOptions<CommonPipelineOptions> {
  Option<bool> printOpGraph{*this, "print-op-graph",
                            llvm::cl::desc("use PrintOpGraphPass."),
                            llvm::cl::init(false)};
  Option<bool> ReduceConstant{*this, "reduce-const",
                              llvm::cl::desc("Reduce constant aggressive"),
                              llvm::cl::init(true)};
  Option<MODE> runMode{
      *this, "mode", llvm::cl::desc("run mode"),
      llvm::cl::init(MODE::Inference),
      llvm::cl::values(
          clEnumValN(MODE::Inference,
                     mode_to_str(MODE::Inference), ""),
          clEnumValN(MODE::Training,
                     mode_to_str(MODE::Training), ""))};
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
  Option<bool> optInArith{*this, "opt-arith",
                          llvm::cl::desc("optimization in arith dialcet"),
                          llvm::cl::init(true)};
  Option<bool> optInLLVM{*this, "opt-llvm",
                         llvm::cl::desc("optimization in llvm dialcet"),
                         llvm::cl::init(true)};
  Option<TARGET> target{
      *this, "target", llvm::cl::desc("target ir"),
      llvm::cl::init(TARGET::CPU),
      llvm::cl::values(clEnumValN(TARGET::CPU,
                                  target_to_str(TARGET::CPU),
                                  "llvm ir"))};
  Option<unsigned> indexBitWidth{
      *this, "index-width", llvm::cl::desc("index-width"), llvm::cl::init(32)};
};
void buildCommonPipeline(::mlir::OpPassManager &pm,
                         const CommonPipelineOptions &options);
void registerCommonPipeline();

}  // namespace llc::pipleline

#endif  // INCLUDE_LLCOMPILER_PIPELINE_COMMONPIPELINE_H_
