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

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/CommonRewrite.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

using namespace mlir;
using namespace mlir::llh;

//===----------------------------------------------------------------------===//
// MaxOp.
//===----------------------------------------------------------------------===//
void MaxOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<MaxOp>>(context);
  results.add<SimplyBinaryOpReshape<MaxOp>>(context, 2);
}

//===----------------------------------------------------------------------===//
// MulOp.
//===----------------------------------------------------------------------===//
void MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<MulOp>>(context);
  results.add<SimplyBinaryOpReshape<MulOp>>(context, 2);
}
//===----------------------------------------------------------------------===//
// AddOp.
//===----------------------------------------------------------------------===//
void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<AddOp>>(context);
  results.add<SimplyBinaryOpReshape<AddOp>>(context, 2);
}
//===----------------------------------------------------------------------===//
// SubOp.
//===----------------------------------------------------------------------===//
void SubOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<SubOp>>(context);
  results.add<SimplyBinaryOpReshape<SubOp>>(context, 2);
}
//===----------------------------------------------------------------------===//
// MinOp.
//===----------------------------------------------------------------------===//
void MinOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<MinOp>>(context);
  results.add<SimplyBinaryOpReshape<MinOp>>(context, 2);
}
//===----------------------------------------------------------------------===//
// DivOp.
//===----------------------------------------------------------------------===//
void DivOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<DivOp>>(context);
  results.add<SimplyBinaryOpReshape<DivOp>>(context, 2);
}

//===----------------------------------------------------------------------===//
// DimOp.
//===----------------------------------------------------------------------===//
namespace {
struct DimOpToConst : public LLHOpRewritePattern<DimOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(DimOp op) const final {
    auto input = op.getInput();
    if (!isa<RankedTensorType>(input.getType())) return llvm::failure();
    auto maybe_const_dim = op.getDim();
    if (!llh::isConstIntegerValue(maybe_const_dim)) return llvm::failure();
    auto type = llc::getRankTensorFrom(input);
    auto dim = llh::getConstIntegerValue(maybe_const_dim);
    if (type.isDynamicDim(dim)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(DimOp op, LLHPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto value = llh::getConstIntegerValue(op);
    auto new_op = rewriter.replaceOpWithNewOp<ConstantOp>(
        op, rewriter.getI64IntegerAttr(value));
  }
};
}  // namespace
void DimOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DimOpToConst>(context);
}

//===----------------------------------------------------------------------===//
// AdaptiveAvgPoolOp
//===----------------------------------------------------------------------===//
void AdaptiveAvgPoolOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// SymbolBindOp
//===----------------------------------------------------------------------===//
namespace {
struct SinkSymbolBindOp : public LLHOpRewritePattern<SymbolBindOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(SymbolBindOp op) const final {
    auto operand = op.getOperand();
    auto input = operand.getDefiningOp();
    if (isa<mlir::index::CastUOp, mlir::index::CastSOp>(input))
      return llvm::success();
    return llvm::failure();
  }
  void rewrite(SymbolBindOp op, LLHPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto befor_cast = op.getOperand();
    auto root = befor_cast.getDefiningOp()->getOperand(0);
    auto new_op = rewriter.replaceOpWithNewOp<SymbolBindOp>(
        op, TypeRange{}, ValueRange{root}, op->getAttrDictionary().getValue());
    rewriter.moveOpAfter(new_op, root.getDefiningOp());
  }
};
}  // namespace

void SymbolBindOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<SinkSymbolBindOp>(context);
}
//===----------------------------------------------------------------------===//
// EncodingBindOp
//===----------------------------------------------------------------------===//
namespace {
struct SinkEncodingBindOp : public LLHOpRewritePattern<EncodingBindOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(EncodingBindOp op) const final { return llvm::success(); }
  void rewrite(EncodingBindOp op, LLHPatternRewriter &rewriter) const final {
    auto operand = op.getOperand();
    if (isa<BlockArgument>(operand)) {
      auto block = op->getBlock();
      op->remove();
      block->push_front(op);
    } else {
      rewriter.moveOpAfter(op, op.getOperand().getDefiningOp());
    }
  }
};
}  // namespace

void EncodingBindOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {
  results.add<SinkEncodingBindOp>(context);
}
//===----------------------------------------------------------------------===//
// SymbolBinaryRelationOp
//===----------------------------------------------------------------------===//
namespace {
//如果symbol表示常量就去除
struct RemoveSymbolBinaryRelationIfAllConst
    : public LLHOpRewritePattern<SymbolBinaryRelationOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(SymbolBinaryRelationOp op) const final {
    auto symbol = op.getSymbol();
    if (!SymbolAnalysis::isConst(symbol)) return llvm::failure();
    auto relation_lhs = op.getRelationsLhs();
    if (!SymbolAnalysis::isConst(relation_lhs)) return llvm::failure();
    auto relation_rhs = op.getRelationsRhs();
    if (!SymbolAnalysis::isConst(relation_rhs)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(SymbolBinaryRelationOp op,
               LLHPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};
}  // namespace

void SymbolBinaryRelationOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {
  results.add<RemoveSymbolBinaryRelationIfAllConst>(context);
}

//===----------------------------------------------------------------------===//
// SymbolRelationOp
//===----------------------------------------------------------------------===//
namespace {
// 替换相等的符号
struct ReplaceSymbolIfEquel : public LLHOpRewritePattern<SymbolRelationOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(SymbolRelationOp op) const final {
    if (op.getRelationKind() == SymbolRelation::EQ) return llvm::success();
    return llvm::failure();
  }
  void rewrite(SymbolRelationOp op, LLHPatternRewriter &rewriter) const final {
    auto symbol = op.getSymbol();
    auto relation = op.getRelation();
    auto analysis = SymbolAnalysis::getInstance(op);
    if (symbol.str() != relation.str()) {
      analysis->replaceSymbol(relation, symbol);
      auto old_symbol = analysis->getOrBuildSymbol(relation);
      rewriter.eraseOp(old_symbol);
    }
    rewriter.eraseOp(op);
  }
};

// 如果symbol表示常量就去除
struct RemoveSymbolRelationIfAllConst
    : public LLHOpRewritePattern<SymbolRelationOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(SymbolRelationOp op) const final {
    auto symbol = op.getSymbol();
    if (!SymbolAnalysis::isConst(symbol)) return llvm::failure();
    auto relation = op.getRelation();
    if (!SymbolAnalysis::isConst(relation)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(SymbolRelationOp op, LLHPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

}  // namespace
void SymbolRelationOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {
  results.add<ReplaceSymbolIfEquel>(context);
  results.add<RemoveSymbolRelationIfAllConst>(context);
}

void mlir::llh::populateSymbolCanonicalizePatterns(
    RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  patterns.add<ReplaceSymbolIfEquel>(context);
  patterns.add<RemoveSymbolRelationIfAllConst>(context);
  patterns.add<RemoveSymbolBinaryRelationIfAllConst>(context);
};