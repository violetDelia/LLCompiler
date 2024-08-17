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
//

#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
namespace mlir::llh {
#define GEN_PASS_DEF_TRANSFORMLAYOUTTONHWC
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace mlir;
using namespace mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
mlir::DenseI64ArrayAttr GetTransposePerms(mlir::Value value) {
  auto context = value.getContext();
  auto perms = mlir::DenseI64ArrayAttr::get(context, {0, 1, 2, 3});
  return perms;
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
namespace {
#include "llcompiler/Dialect/LLH/Transforms/TransformLayoutToNHWC.inc"
}
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateTransformLayoutToNHWCPatterns(RewritePatternSet& patterns) {
  populateWithGenerated(patterns);
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct TransformLayoutToNHWC
    : llh::impl::TransformLayoutToNHWCBase<TransformLayoutToNHWC> {
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void TransformLayoutToNHWC::runOnOperation() {
  auto* context = &getContext();
  RewritePatternSet patterns(context);
  populateTransformLayoutToNHWCPatterns(patterns);
  auto op = getOperation();
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// pass create
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> mlir::llh::createTransformLayoutToNHWCPass() {
  return std::make_unique<TransformLayoutToNHWC>();
}