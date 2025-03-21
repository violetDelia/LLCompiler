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

#include "llcompiler/Dialect/Utility/Benefit.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class LLHPatternRewriter : public RewriterBase {
 public:
  explicit LLHPatternRewriter(MLIRContext *ctx) : RewriterBase(ctx) {}
  explicit LLHPatternRewriter(Operation *op) : RewriterBase(op) {}
  using RewriterBase::RewriterBase;

  virtual void processWileBuildOperation(Operation *op);

  virtual bool canRecoverFromRewriteFailure() const;

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
    auto newOp =
        RewriterBase::create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
    replaceOp(op, newOp.getOperation());
    processWileBuildOperation(newOp);
    return newOp;
  }
};

namespace detail {
template <typename SourceOp>
struct LLCOpOrInterfaceRewritePatternBase : public RewritePattern {
  using RewritePattern::RewritePattern;

  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), rewriter);
  }

  LogicalResult match(Operation *op) const final {
    auto result = match(cast<SourceOp>(op));
    return result;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    DEBUG(llc::MLIR_PASS);
    DEBUG(llc::MLIR_PASS) << "run in pattern " << this->getDebugName().str();
    auto llh_rewriter = LLHPatternRewriter(rewriter.getContext());
    llh_rewriter.setInsertionPoint(rewriter.getBlock(),
                                   rewriter.getInsertionPoint());
    return matchAndRewrite(cast<SourceOp>(op), llh_rewriter);
  }

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
      DEBUG(llc::MLIR_PASS)
          << "rewrite " << cast<SourceOp>(op).getOperationName().str()
          << " in pattern " << this->getDebugName().str();
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
struct LLHOpRewritePattern
    : public detail::LLCOpOrInterfaceRewritePatternBase<SourceOp> {
  /// Patterns must specify the root operation name they match against, and
  /// can also specify the benefit of the pattern matching and a list of
  /// generated ops.
  LLHOpRewritePattern(MLIRContext *context, PatternBenefit benefit = 1,
                      ArrayRef<StringRef> generatedNames = {})
      : detail::LLCOpOrInterfaceRewritePatternBase<SourceOp>(
            SourceOp::getOperationName(), benefit, context, generatedNames) {}
};

template <class SourceOp, class TargetOp>
struct SimplyFullLowing : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult match(SourceOp op) const final { return success(); }

  void rewrite(SourceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto types = op->getResultTypes();
    auto operands = op->getOperands();
    auto attrs = op->getAttrDictionary().getValue();
    auto new_op = rewriter.create<TargetOp>(loc, types, operands, attrs);
    rewriter.replaceOp(op, new_op);
  }
};

template <class Op>
struct EraseNoUserOp : public LLHOpRewritePattern<Op> {
  explicit EraseNoUserOp(MLIRContext *context,
                         PatternBenefit benefit = llh::RemoveBenfit,
                         ArrayRef<StringRef> generatedNames = {})
      : LLHOpRewritePattern<Op>(context, benefit, generatedNames){};
  LogicalResult match(Op op) const final {
    if (op->getUsers().empty()) return llvm::success();
    return llvm::failure();
  }
  void rewrite(Op op, LLHPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};
}  // namespace mlir
#endif  // INCLUDE_LLCOMPILER_DIALECT_UTILITY_REWRITEPATTERN_H_
