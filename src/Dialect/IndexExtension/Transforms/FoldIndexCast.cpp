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

#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::index::ex {
#define GEN_PASS_DEF_FOLDINDEXCASTPASS
#include "llcompiler/Dialect/IndexExtension/Transforms/Passes.h.inc"
}  // namespace mlir::index::ex
using namespace ::mlir;
using namespace ::mlir::index;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
template <class CastOp>
struct FoldCastOp : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp op, PatternRewriter& rewriter) const {
    if (!isa<CastOp>(op->getOperand(0).getDefiningOp())) return llvm::failure();
    auto type = op->getResult(0).getType();
    auto front_cast_res = op->getOperand(0);
    auto front_cast = front_cast_res.getDefiningOp();
    auto front_type = front_cast->getOperand(0).getType();
    if (front_type != type) return llvm::failure();
    rewriter.replaceAllUsesWith(op->getResult(0), front_cast->getOperand(0));
    return llvm::success();
  }
};

struct ConstOpToArith : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult match(ConstantOp op) const { return llvm::success(); }

  void rewrite(ConstantOp op, PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
        op, *op.getValue().getRawData());
  }
};

struct FoldFromElements : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern<tensor::FromElementsOp>::OpRewritePattern;

  LogicalResult match(tensor::FromElementsOp op) const {
    auto res_type = op.getResult().getType();
    auto ele_type = res_type.getElementType();
    if (isa<IndexType>(ele_type)) return llvm::failure();
    auto operands = op.getElements();
    for (auto operand : operands) {
      if (!isa<CastUOp, CastSOp, arith::ConstantOp, arith::IndexCastOp>(
              operand.getDefiningOp()))
        return llvm::failure();
      if (isa<CastUOp, CastSOp, arith::IndexCastOp>(operand.getDefiningOp())) {
        auto type = operand.getDefiningOp()->getOperand(0).getType();
        if (!isa<IndexType>(type)) return llvm::failure();
      }
    }

    return llvm::success();
  }

  void rewrite(tensor::FromElementsOp op, PatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    auto elements = op.getElements();
    llvm::SmallVector<Value, 0> new_elements;
    for (auto operand : elements) {
      if (isa<CastUOp, CastSOp,arith::IndexCastOp>(operand.getDefiningOp())) {
        new_elements.push_back(operand.getDefiningOp()->getOperand(0));
      } else if (isa<arith::ConstantOp>(operand.getDefiningOp())) {
        auto const_op = cast<arith::ConstantOp>(operand.getDefiningOp());
        auto value = const_op.getValue();
        CHECK(llc::MLIR_PASS, isa<IntegerAttr>(value));
        auto int_value = cast<IntegerAttr>(value);
        auto new_const = rewriter.create<arith::ConstantIndexOp>(
            loc, *int_value.getValue().getRawData());
        new_elements.push_back(new_const);
      } else if (isa<arith::ConstantOp>(operand.getDefiningOp())) {
      } else {
        operand.dump();
        UNIMPLEMENTED(llc::MLIR_PASS)
            << operand.getDefiningOp()->getName().getStringRef().str();
      }
    }
    auto new_op = rewriter.create<tensor::FromElementsOp>(loc, new_elements);
    rewriter.replaceAllUsesWith(op, new_op);
    rewriter.replaceOp(op, new_op);
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateFoldIndexCastPassPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<FoldCastOp<CastSOp>>(context, 2);
  patterns.add<FoldCastOp<CastUOp>>(context, 2);
  patterns.add<ConstOpToArith>(context);
  patterns.add<FoldFromElements>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct FoldIndexCastPass
    : ::mlir::index::ex::impl::FoldIndexCastPassBase<FoldIndexCastPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void FoldIndexCastPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateFoldIndexCastPassPassPatterns(patterns);
  auto op = getOperation();
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
