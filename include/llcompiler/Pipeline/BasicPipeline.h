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

#ifndef INCLUDE_LLCOMPILER_PIPELINE_BASICPIPELINE_H_
#define INCLUDE_LLCOMPILER_PIPELINE_BASICPIPELINE_H_
#include <cstddef>
#include <cstdint>
#include <string>

#include "llcompiler/Support/Enums.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace llc::pipleline {
struct BasicPipelineOptions
    : public mlir::PassPipelineOptions<BasicPipelineOptions> {
  Option<MODE> runMode{
      *this, "mode", llvm::cl::desc("run mode"),
      llvm::cl::init(MODE::Inference),
      llvm::cl::values(
          clEnumValN(MODE::Inference, mode_to_str(MODE::Inference), ""),
          clEnumValN(MODE::Training, mode_to_str(MODE::Training), ""))};
  Option<bool> symbolInfer{*this, "symbol-infer",
                            llvm::cl::desc("symbol-infer"),
                            llvm::cl::init(false)};
  Option<TARGET> target{*this, "target", llvm::cl::desc("target ir"),
                        llvm::cl::init(TARGET::CPU),
                        llvm::cl::values(clEnumValN(
                            TARGET::CPU, target_to_str(TARGET::CPU), "cpu"))};
  Option<std::string> irTreeDir{
      *this, "ir_tree_dir", llvm::cl::desc("ir tree dir"), llvm::cl::init("")};
  Option<unsigned> indexBitWidth = {*this, "index bit width",
                                   llvm::cl::desc("index bit width"),
                                   llvm::cl::init(32)};
                    
};
void buildBasicPipeline(mlir::OpPassManager &pm,
                        const BasicPipelineOptions &options);
void registerBasicPipeline();

}  // namespace llc::pipleline
#endif  // INCLUDE_LLCOMPILER_PIPELINE_BASICPIPELINE_H_
