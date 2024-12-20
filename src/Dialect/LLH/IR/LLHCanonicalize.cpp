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
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/CommonRewrite.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Benefit.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
using namespace mlir;
using namespace mlir::llh;
#include "llcompiler/Dialect/LLH/IR/LLHCanonicalize.inc"

//===----------------------------------------------------------------------===//
// ConstantOp.
//===----------------------------------------------------------------------===//
void ConstantOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<EraseNoUserOp<ConstantOp>>(context, RemoveBenfit);
}
//===----------------------------------------------------------------------===//
// AbsOp.
//===----------------------------------------------------------------------===//
void AbsOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<FoldTwoAbsOpPattern>(context);
  results.add<EraseNoUserOp<AbsOp>>(context, RemoveBenfit);
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
    auto value = llh::getConstIntegerValue(op);
    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            rewriter.getI64IntegerAttr(value));
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
    auto new_index = rewriter.create<AddOp>(loc, rewriter.getI64Type(),
                                            ValueRange{dim, index});
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

//===----------------------------------------------------------------------===//
// ReShapeOp.
//===----------------------------------------------------------------------===//
namespace {
struct FoldReshapeOp : public LLHOpRewritePattern<ReshapeOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(ReshapeOp op) const final {
    auto res = op.getResult();
    auto tensor = llc::getRankTensorFrom(res);
    if (!tensor.hasStaticShape()) return llvm::failure();
    auto input = op.getInput();
    if (isa<BlockArgument>(input)) return llvm::failure();
    if (!isa<ConstantOp>(input.getDefiningOp())) return llvm::failure();
    return llvm::success();
  }
  void rewrite(ReshapeOp op, LLHPatternRewriter &rewriter) const final {
    auto res = op.getResult();
    auto type = llc::getRankTensorFrom(res);
    auto const_op = llvm::dyn_cast<ConstantOp>(op.getInput().getDefiningOp());
    auto value = llvm::dyn_cast_or_null<DenseElementsAttr>(const_op.getValue());
    CHECK(llc::MLIR, value);
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto new_value =
        DenseElementsAttr::getFromRawBuffer(new_type, value.getRawData());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, new_value);
  }
};
}  // namespace
void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<FoldReshapeOp>(context);
  results.add<EraseNoUserOp<ReshapeOp>>(context);
}

//===----------------------------------------------------------------------===//
// BroadCastToOp.
//===----------------------------------------------------------------------===//
namespace {
struct FoldBroadCastToOp : public LLHOpRewritePattern<BroadCastToOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(BroadCastToOp op) const final {
    auto maybe_noexpand_dims = op.getNoexpandDims();
    if (!maybe_noexpand_dims.has_value()) return llvm::failure();
    auto noexpand_dims = maybe_noexpand_dims.value();
    for (int i = 0; i < noexpand_dims.size(); ++i) {
      if (noexpand_dims[i] != i) return llvm::failure();
    }
    if (noexpand_dims.size() != op.getOutShapes().size())
      return llvm::failure();
    return llvm::success();
  }
  void rewrite(BroadCastToOp op, LLHPatternRewriter &rewriter) const final {
    auto input = op.getInput();
    rewriter.replaceAllUsesWith(op.getResult(), input);
  }
};
}  // namespace
void BroadCastToOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {
  results.add<FoldBroadCastToOp>(context);
  results.add<EraseNoUserOp<BroadCastToOp>>(context);
}
