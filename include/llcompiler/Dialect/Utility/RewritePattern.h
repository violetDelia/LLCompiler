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
#ifndef INCLUDE_LLCOMPILER_DIALECT_UTILITY_REWRITEPATTERN_H_
#define INCLUDE_LLCOMPILER_DIALECT_UTILITY_REWRITEPATTERN_H_
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

class LLHPatternRewriter : public PatternRewriter {
  using PatternRewriter::create;

 public:
  explicit LLHPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  using PatternRewriter::PatternRewriter;

  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    OperationState state(location,
                         getCheckRegisteredInfo<OpTy>(location.getContext()));
    OpTy::build(*this, state, std::forward<Args>(args)...);
    auto *op = create(state);
    auto result = dyn_cast<OpTy>(op);
    assert(result && "builder didn't return the right type");
    DINFO << "symbol not supported";
    return result;
  }

  template <typename OpTy, typename... Args>
  void createOrFold(SmallVectorImpl<Value> &results, Location location,
                    Args &&...args) {
    OperationState state(location,
                         getCheckRegisteredInfo<OpTy>(location.getContext()));
    OpTy::build(*this, state, std::forward<Args>(args)...);
    Operation *op = Operation::create(state);
    DINFO << "symbol not supported";
    auto block = this->getBlock();
    auto insertPoint = this->getInsertionPoint();
    if (block) block->getOperations().insert(insertPoint, op);
    if (succeeded(tryFold(op, results)) && !results.empty()) {
      op->erase();
      return;
    }
    ResultRange opResults = op->getResults();
    results.assign(opResults.begin(), opResults.end());
    if (block && listener)
      listener->notifyOperationInserted(op, /*previous=*/{});
  }
};

namespace detail {
template <typename SourceOp>
struct LLCOpOrInterfaceRewritePatternBase : public RewritePattern {
  using RewritePattern::RewritePattern;

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), rewriter);
    DEBUG(llc::MLIR_PASS) << "rewrite "
                          << cast<SourceOp>(op).getOperationName().str()
                          << " in pattern " << this->getDebugName().str();
  }
  LogicalResult match(Operation *op) const final {
    DEBUG(llc::MLIR_PASS) << "run in pattern " << this->getDebugName().str();
    auto result = match(cast<SourceOp>(op));
    return result;
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    auto llh_rewriter = LLHPatternRewriter(rewriter.getContext());
    return matchAndRewrite(cast<SourceOp>(op), llh_rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, LLHPatternRewriter &rewriter) const {
    llvm_unreachable("must override rewrite or matchAndRewrite");
  }
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        LLHPatternRewriter &rewriter) const {
    if (succeeded(match(op))) {
      rewrite(op, rewriter);
      return success();
    }
    DEBUG(llc::MLIR_PASS) << "match pattern failed"
                          << this->getDebugName().str();
    return failure();
  }
};
}  // namespace detail

/// OpRewritePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp>
struct LLCOpRewritePattern
    : public detail::LLCOpOrInterfaceRewritePatternBase<SourceOp> {
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching and a list of generated
  /// ops.
  LLCOpRewritePattern(MLIRContext *context, PatternBenefit benefit = 1,
                      ArrayRef<StringRef> generatedNames = {})
      : detail::LLCOpOrInterfaceRewritePatternBase<SourceOp>(
            SourceOp::getOperationName(), benefit, context, generatedNames) {}
};
}  // namespace mlir
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_REWRITEPATTERN_H_
