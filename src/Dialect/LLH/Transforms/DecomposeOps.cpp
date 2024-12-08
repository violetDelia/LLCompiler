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
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/InferSymbol.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_DECOMPOSEOPSPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
ConstantOp genOneTenorConst(LLHPatternRewriter& rewriter, Type float_type,
                            FloatAttr value, Location loc) {
  auto double_value = value.getValue().convertToDouble();
  auto new_value = rewriter.getFloatAttr(float_type, double_value).getValue();
  auto one_tensor = RankedTensorType::get({1}, float_type);
  auto value_attr = DenseElementsAttr::get(one_tensor, {new_value});
  auto const_op = rewriter.create<ConstantOp>(loc, value_attr);
  return const_op;
}

BroadCastToOp reshapeAndBroadcastTo(LLHPatternRewriter& rewriter,
                                    int64_t feature_index, Value input,
                                    Value value, Location loc) {
  auto input_type = llc::getRankTensorFrom(input);
  auto rank = input_type.getRank();
  auto one_const = rewriter.create<ConstantOp>(
      loc, IntegerAttr::get(rewriter.getI64Type(), 1));
  auto reshape_shapes = llvm::SmallVector<int64_t>();
  auto reshape_dims = llvm::SmallVector<mlir::Value>();
  for (auto i = 0; i < rank; ++i) {
    if (i != feature_index) {
      reshape_dims.push_back(one_const);
      reshape_shapes.push_back(1);
    } else {
      auto feature_dim = llh::buildTensorDim(input, &rewriter, i);
      reshape_dims.push_back(feature_dim);
      reshape_shapes.push_back(input_type.getDimSize(i));
    }
  }
  auto reshape_type = input_type.clone(reshape_shapes);

  auto cast_dims = llvm::SmallVector<int64_t>();
  for (auto i = 0; i < rank; ++i) {
    if (i != feature_index) cast_dims.push_back(i);
  }
  auto dims = llh::buildTensorDims(input, &rewriter);

  auto reshape =
      rewriter.create<ReshapeOp>(loc, reshape_type, value, reshape_dims);
  auto broadcast_to =
      rewriter.create<BroadCastToOp>(loc, input_type, reshape, dims, cast_dims);
  return broadcast_to;
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct DecomposeBatchNormOp : public LLHOpRewritePattern<BatchNormOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp op,
                                LLHPatternRewriter& rewriter) const final {
    auto val = op.getInputVar();
    auto val_type = llc::getRankTensorFrom(val);
    auto float_type = val_type.getElementType();
    if (!isa<FloatType>(float_type)) return llvm::failure();
    auto loc = op->getLoc();
    auto input = op.getInput();
    auto input_type = llc::getRankTensorFrom(input);
    auto epsilon = op.getEpsilonAttr();
    auto feature_index = op.getFeatureIndex();
    auto epsilon_const = genOneTenorConst(rewriter, float_type, epsilon, loc);
    auto fined_val = rewriter.create<AddOp>(loc, TypeRange{val.getType()},
                                            ValueRange{val, epsilon_const});
    auto stddev = rewriter.create<SqrtOp>(loc, fined_val.getType(), fined_val);

    auto mean = reshapeAndBroadcastTo(rewriter, feature_index, input,
                                      op.getInputMean(), loc);
    auto scale = reshapeAndBroadcastTo(rewriter, feature_index, input,
                                       op.getScale(), loc);

    auto bias = reshapeAndBroadcastTo(rewriter, feature_index, input,
                                      op.getBias(), loc);
    auto broadcast_stddev =
        reshapeAndBroadcastTo(rewriter, feature_index, input, stddev, loc);

    // scale * (input - mean) / stddev + bias
    Value result;
    result = rewriter.create<mlir::llh::SubOp>(loc, TypeRange{input_type},
                                               ValueRange{input, mean});
    result = rewriter.create<MulOp>(loc, TypeRange{input_type},
                                    ValueRange{result, scale});
    result = rewriter.create<DivOp>(loc, TypeRange{input_type},
                                    ValueRange{result, broadcast_stddev});
    rewriter.replaceOpWithNewOp<AddOp>(op, TypeRange{input_type},
                                       ValueRange{result, bias});
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateDecomposeOpsPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<DecomposeBatchNormOp>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct DecomposeOpsPass : llh::impl::DecomposeOpsPassBase<DecomposeOpsPass> {
  using DecomposeOpsPassBase::DecomposeOpsPassBase;
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void DecomposeOpsPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateDecomposeOpsPassPatterns(patterns);
  populateSymbolCanonicalizePatterns(patterns);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();

  LLC_RUN_OUT_PASS
}
