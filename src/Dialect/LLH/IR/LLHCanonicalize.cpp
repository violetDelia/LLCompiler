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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::llh;
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

void AdaptiveAvgPoolOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {}
void SymbolBindOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                               MLIRContext *context) {}
void DimOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                        MLIRContext *context) {}
void EncodingBindOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, MLIRContext *context) {}