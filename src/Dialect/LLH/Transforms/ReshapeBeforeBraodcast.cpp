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

#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Interfaces/BraodcastableOpInterfaces.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_RESHAPEBEFOREBRAODCASTPASS
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

struct SimplyBinaryOp : public LLHOpRewritePattern<AddOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(AddOp op) const final {
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);
    auto lhs_rank = llc::getRankTensorFrom(lhs).getRank();
    auto rhs_rank = llc::getRankTensorFrom(rhs).getRank();
    if (lhs_rank == rhs_rank) return llvm::failure();
    return llvm::success();
  }
  void rewrite(AddOp op, LLHPatternRewriter& rewriter) const final {
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);
    auto lhs_tensor = llc::getRankTensorFrom(lhs);
    auto rhs_tensor = llc::getRankTensorFrom(rhs);
    auto lhs_rank = lhs_tensor.getRank();
    auto rhs_rank = rhs_tensor.getRank();
    Value higher_value, lower_value;
    if (lhs_rank > rhs_rank) {
      higher_value = lhs;
      lower_value = rhs;
    } else {
      higher_value = rhs;
      lower_value = lhs;
    }
    op->dump();
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateReshapeBeforeBraodcastPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<SimplyBinaryOp>(context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configReshapeBeforeBraodcastPassConversionTarget(
    ConversionTarget& target) {
  // target.addIllegalOp<llh::SymbolicBindOp>();
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct ReshapeBeforeBraodcastPass
    : llh::impl::ReshapeBeforeBraodcastPassBase<ReshapeBeforeBraodcastPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void ReshapeBeforeBraodcastPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateReshapeBeforeBraodcastPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
