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
#include <cstdio>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/CommonRewrite.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "llcompiler/Dialect/LLH/IR/LLHCanonicalize.inc"
constexpr inline size_t SinkOpBenfit = 101;
constexpr inline size_t RefineOpBenefit = 101;
constexpr inline size_t ReshapeBenefit = 100;
constexpr inline size_t BroadcastBenefit = 99;

//===----------------------------------------------------------------------===//
// ConstantOp.
//===----------------------------------------------------------------------===//
void ConstantOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<EraseNoUserOp<ConstantOp>>(context);
}
//===----------------------------------------------------------------------===//
// AbsOp.
//===----------------------------------------------------------------------===//
void AbsOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldTwoAbsOpPattern>(context);
  results.add<EraseNoUserOp<AbsOp>>(context);
}
//===----------------------------------------------------------------------===//
// MaxOp.
//===----------------------------------------------------------------------===//
void MaxOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<MaxOp>>(context, BroadcastBenefit);
  results.add<SimplyBinaryOpReshape<MaxOp>>(context, ReshapeBenefit);
}

//===----------------------------------------------------------------------===//
// MulOp.
//===----------------------------------------------------------------------===//
void MulOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<MulOp>>(context, BroadcastBenefit);
  results.add<SimplyBinaryOpReshape<MulOp>>(context, ReshapeBenefit);
}
//===----------------------------------------------------------------------===//
// AddOp.
//===----------------------------------------------------------------------===//
void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<AddOp>>(context, BroadcastBenefit);
  results.add<SimplyBinaryOpReshape<AddOp>>(context, ReshapeBenefit);
}
//===----------------------------------------------------------------------===//
// SubOp.
//===----------------------------------------------------------------------===//
void SubOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<SubOp>>(context, BroadcastBenefit);
  results.add<SimplyBinaryOpReshape<SubOp>>(context, ReshapeBenefit);
}
//===----------------------------------------------------------------------===//
// MinOp.
//===----------------------------------------------------------------------===//
void MinOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<MinOp>>(context, BroadcastBenefit);
  results.add<SimplyBinaryOpReshape<MinOp>>(context, ReshapeBenefit);
}
//===----------------------------------------------------------------------===//
// DivOp.
//===----------------------------------------------------------------------===//
void DivOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<SimplyBinaryOpInsertBraodcast<DivOp>>(context, BroadcastBenefit);
  results.add<SimplyBinaryOpReshape<DivOp>>(context, ReshapeBenefit);
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
struct MoveSymbolBindOp : public LLHOpRewritePattern<SymbolBindOp> {
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

struct SinkSymbolBindOp : public LLHOpRewritePattern<SymbolBindOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult match(SymbolBindOp op) const final {
    auto input = op.getOperand();
    if (!isa<mlir::arith::IndexCastOp>(input.getDefiningOp()))
      return llvm::failure();
    return llvm::success();
  }
  void rewrite(SymbolBindOp op, LLHPatternRewriter &rewriter) const final {
    auto operand = op.getOperand().getDefiningOp();
    op->setOperand(0, operand->getOperand(0));
  }
};
}  // namespace

void SymbolBindOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<SinkSymbolBindOp>(context, SinkOpBenfit);
  results.add<MoveSymbolBindOp>(context);
}
//===----------------------------------------------------------------------===//
// EncodingBindOp
//===----------------------------------------------------------------------===//
namespace {
struct MoveEncodingBindOp : public LLHOpRewritePattern<EncodingBindOp> {
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
  results.add<MoveEncodingBindOp>(context);
}
//===----------------------------------------------------------------------===//
// SymbolBinaryRelationOp
//===----------------------------------------------------------------------===//
namespace {
// 如果symbol表示常量就去除
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
    if (op.getRelationKind() != SymbolRelation::EQ) return llvm::failure();
    auto analysis = SymbolAnalysis::getInstance(op);
    auto symbol = op.getSymbol();
    if (analysis->isConst(symbol)) return llvm::failure();
    auto relation = op.getRelation();
    if (analysis->isConst(relation)) return llvm::failure();
    return llvm::success();
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

//===----------------------------------------------------------------------===//
// Layout Pattern
//===----------------------------------------------------------------------===//
namespace {
template <class OP>
struct AddConvLayoutAttr : public LLHOpRewritePattern<OP> {
  using LLHOpRewritePattern<OP>::LLHOpRewritePattern;

  LogicalResult match(OP op) const final {
    auto module = op->template getParentOfType<ModuleOp>();
    CHECK(llc::MLIR_PASS, module->hasAttr(llc::GloabalLayoutAttr));
    if (!module->hasAttr(llc::GloabalLayoutAttr)) return llvm::failure();
    if (!op->hasAttr(llc::LayoutAttr)) return llvm::success();
    if (!op->hasAttr(llc::WeightLayoutAttr)) return llvm::success();
    return llvm::failure();
  }

  void rewrite(OP op, LLHPatternRewriter &rewriter) const final {
    auto context = op->getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    auto global_layout = module->getAttr(llc::GloabalLayoutAttr);
    CHECK(llc::MLIR_PASS, llvm::isa<StringAttr>(global_layout));
    auto maybe_layout =
        symbolizeLayout(dyn_cast<StringAttr>(global_layout).getValue());
    CHECK(llc::MLIR_PASS, maybe_layout.has_value());
    auto layout = maybe_layout.value();
    auto tensor = llc::getRankTensorFrom(op);
    auto rank = tensor.getRank();
    auto input_layout = llh::getLayoutFromGloabalLayout(layout, rank);
    auto weight_layout = llh::getWeightLayoutFromGloabalLayout(layout, rank);
    llc::add_layout_attr(op, input_layout);
    llc::add_weight_layout_attr(op, weight_layout);
  }
};

template <class OP>
struct AddLayoutAttr : public LLHOpRewritePattern<OP> {
  using LLHOpRewritePattern<OP>::LLHOpRewritePattern;

  LogicalResult match(OP op) const final {
    auto module = op->template getParentOfType<ModuleOp>();
    CHECK(llc::MLIR_PASS, module->hasAttr(llc::GloabalLayoutAttr));
    if (!module->hasAttr(llc::GloabalLayoutAttr)) return llvm::failure();
    if (!op->hasAttr(llc::LayoutAttr)) return llvm::success();
    return llvm::failure();
  }

  void rewrite(OP op, LLHPatternRewriter &rewriter) const final {
    auto context = op->getContext();
    auto module = op->template getParentOfType<ModuleOp>();
    auto global_layout = module->getAttr(llc::GloabalLayoutAttr);
    CHECK(llc::MLIR_PASS, llvm::isa<StringAttr>(global_layout));
    auto maybe_layout =
        symbolizeLayout(dyn_cast<StringAttr>(global_layout).getValue());
    CHECK(llc::MLIR_PASS, maybe_layout.has_value());
    auto layout = maybe_layout.value();
    auto tensor = llc::getRankTensorFrom(op);
    auto rank = tensor.getRank();
    auto input_layout = llh::getLayoutFromGloabalLayout(layout, rank);
    llc::add_layout_attr(op, input_layout);
  }
};
}  // namespace
//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//
void ConvOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<AddConvLayoutAttr<ConvOp>>(context);
}

//===----------------------------------------------------------------------===//
// ConvBaisOp
//===----------------------------------------------------------------------===//
void ConvBiasOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<AddConvLayoutAttr<ConvBiasOp>>(context);
}
//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//
void MaxPoolOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<AddLayoutAttr<MaxPoolOp>>(context);
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//
namespace {
struct ExtractOpRefine : public LLHOpRewritePattern<ExtractOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(ExtractOp op) const final {
    auto index = op.getIndex();
    if (!llh::isConstIntegerValue(index)) return llvm::failure();
    auto index_value = llh::getConstIntegerValue(index);
    if (index_value >= 0) return llvm::failure();
    return llvm::success();
  }
  void rewrite(ExtractOp op, LLHPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto index = op.getIndex();
    auto input = op.getInput();
    auto dim = rewriter.create<DimOp>(loc, input, 0);
    auto index_value = llh::getConstIntegerValue(index);
    auto offset = rewriter.create<ConstantOp>(
        loc, rewriter.getI64IntegerAttr(-index_value - 1));
    auto new_index = rewriter.create<SubOp>(loc, rewriter.getI64Type(),
                                            ValueRange{dim, offset});
    op->setOperand(1, new_index);
  }
};

struct ExtractOpToSliceOp : public LLHOpRewritePattern<ExtractOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(ExtractOp op) const final { return llvm::success(); }

  void rewrite(ExtractOp op, LLHPatternRewriter &rewriter) const final {}
};
}  // namespace
void ExtractOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ExtractOpRefine>(context, RefineOpBenefit);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//
void SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          MLIRContext *context) {}

void mlir::llh::populateSymbolCanonicalizePatterns(
    RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  patterns.add<ReplaceSymbolIfEquel>(context);
  patterns.add<RemoveSymbolRelationIfAllConst>(context);
  patterns.add<RemoveSymbolBinaryRelationIfAllConst>(context);
  patterns.add<SinkSymbolBindOp>(context, SinkOpBenfit);
}
