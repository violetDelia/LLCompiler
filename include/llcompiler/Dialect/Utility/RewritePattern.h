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
#include <utility>

#include "llcompiler/Support/Logger.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

namespace mlir {

class LLCPatternRewriter : public RewriterBase {
 public:
  explicit LLCPatternRewriter(MLIRContext *ctx) : RewriterBase(ctx) {}
  using RewriterBase::RewriterBase;

  virtual void processWileBuildOperation(Operation *op) {
    // DINFO << "build: " << op->getName().getStringRef().str();
  }

  virtual bool canRecoverFromRewriteFailure() const { return false; }

 public:
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    auto op = RewriterBase::create<OpTy>(location, std::forward<Args>(args)...);
    processWileBuildOperation(op);
    return op;
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value>
  createOrFold(Location location, Args &&...args) {
    SmallVector<Value, 1> results;
    RewriterBase::createOrFold<OpTy>(results, location,
                                     std::forward<Args>(args)...);
    auto op = results.front().getDefiningOp();
    processWileBuildOperation(op);
    return results.front();
  }

  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy>
  createOrFold(Location location, Args &&...args) {
    auto op = create<OpTy>(location, std::forward<Args>(args)...);
    SmallVector<Value, 0> unused;
    (void)tryFold(op.getOperation(), unused);
    return op;
  }

  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(Operation *op, Args &&...args) {
    auto newOp = create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOp(op, newOp.getOperation());
    return newOp;
  }
};

namespace detail {
template <typename SourceOp>
struct LLCOpOrInterfaceRewritePatternBase : public RewritePattern {
  using RewritePattern::RewritePattern;

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
    auto llh_rewriter = LLCPatternRewriter(rewriter.getContext());
    llh_rewriter.setInsertionPoint(rewriter.getBlock(),
                                   rewriter.getInsertionPoint());
    return matchAndRewrite(cast<SourceOp>(op), llh_rewriter);
  }

  virtual void rewrite(SourceOp op, LLCPatternRewriter &rewriter) const {
    llvm_unreachable("must override rewrite or matchAndRewrite");
  }
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        LLCPatternRewriter &rewriter) const {
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
