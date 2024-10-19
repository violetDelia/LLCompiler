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
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::llh;
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
  }
};
}  // namespace
void DimOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// AdaptiveAvgPoolOp
//===----------------------------------------------------------------------===//
void AdaptiveAvgPoolOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// SymbolBindOp
//===----------------------------------------------------------------------===//
void SymbolBindOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               MLIRContext *context) {}
//===----------------------------------------------------------------------===//
// EncodingBindOp
//===----------------------------------------------------------------------===//

void EncodingBindOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {}
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