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
#include "Dialect/LLH/IR/LLHEnums.h"
#include "Dialect/Utility/Attribute.h"
#include "Support/Enums.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace llc::pipeline {
struct TransformPipelineOptions
    : public mlir::PassPipelineOptions<TransformPipelineOptions> {
  Option<bool> symbolInfer{*this, "symbol-infer",
                           llvm::cl::desc("symbol-infer"),
                           llvm::cl::init(false)};
  Option<TARGET> target{*this, "target", llvm::cl::desc("target ir"),
                        llvm::cl::init(TARGET::CPU),
                        llvm::cl::values(clEnumValN(
                            TARGET::CPU, target_to_str(TARGET::CPU), "cpu"))};
  Option<uint64_t> L3CacheSize = {*this, "L3 bytes size",
                                  llvm::cl::desc("L3 bytes size"),
                                  llvm::cl::init(37748736)};
  Option<uint64_t> L2CacheSize = {*this, "L2 bytes size",
                                  llvm::cl::desc("L2 bytes size"),
                                  llvm::cl::init(2097152)};
  Option<uint64_t> L1CacheSize = {*this, "L1 bytes size",
                                  llvm::cl::desc("L1 bytes size"),
                                  llvm::cl::init(49152)};
  Option<mlir::llh::Layout> targetLayout = {
      *this, "layout", llvm::cl::desc("layout"),
      llvm::cl::init(mlir::llh::Layout::NHWC)};

  ListOption<std::string> transformLibraryPaths = {
      *this, "transform-library-path", llvm::cl::desc("transformLibraryPaths")};

  ListOption<std::string> transformEntryPoint = {
      *this, "transform-entry-point", llvm::cl::desc("transformEntryPoint")};
  Option<unsigned> indexBitWidth = {*this, "index bit width",
                                    llvm::cl::desc("index bit width"),
                                    llvm::cl::init(32)};
};
void buildTransformPipeline(mlir::OpPassManager &pm,
                            const TransformPipelineOptions &options);
void registerTransformPipeline();

}  // namespace llc::pipeline

#endif  // INCLUDE_LLCOMPILER_PIPELINE_COMMONPIPELINE_H_
