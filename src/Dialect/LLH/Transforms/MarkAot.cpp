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

#include <cstdint>

#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_MARKAOTPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateMarkAotPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
}
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct MarkAotPass : llh::impl::MarkAotPassBase<MarkAotPass> {
  void runOnOperation() override;
};

}  // namespace
void MarkAotPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  populateMarkAotPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
