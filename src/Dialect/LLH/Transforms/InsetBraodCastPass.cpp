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

#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Interfaces/BraodcastableOpInterfaces.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_INSERTBROADCASTPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//

template <class BinaryOp>
struct SimplyBinaryOp : public LLHOpRewritePattern<BinaryOp> {
  using LLHOpRewritePattern<BinaryOp>::LLHOpRewritePattern;
  LogicalResult match(BinaryOp op) const final {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    if (!isa<RankedTensorType>(lhs.getType())) return ::failure();
    auto lhs_type = llc::getRankTensorFrom(lhs);
    auto rhs_type = llc::getRankTensorFrom(rhs);
    if (lhs_type == rhs_type) return llvm::failure();
    return llvm::success();
  }
  void rewrite(BinaryOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhs_type = llc::getRankTensorFrom(lhs);
    auto rhs_type = llc::getRankTensorFrom(rhs);
    auto result = op->getResult(0);
    auto result_type = llc::getRankTensorFrom(result);
    Value will_be_broadcast;
    Value target_operand;
    if (lhs_type == result_type) {
      will_be_broadcast = rhs;
      target_operand = lhs;

    } else if (rhs_type == result_type) {
      will_be_broadcast = lhs;
      target_operand = rhs;
    } else {
      op.dump();
      FATAL(llc::MLIR_PASS) << "Unexpected result";
    }
    auto before_braodcast_type = llc::getRankTensorFrom(will_be_broadcast);
    auto target_type = result_type;
    llvm::SmallVector<int64_t> cast_dims;
    auto before_encode = llc::getEncodingFrom(before_braodcast_type);
    auto target_encode = llc::getEncodingFrom(target_type);
    auto before_symbol = before_encode.getShapeSymbols();
    auto target_symbol = target_encode.getShapeSymbols();
    for (size_t i = 0; i < result_type.getRank(); i++) {
      if (before_symbol[i] != target_symbol[i]) {
        cast_dims.push_back(i);
      }
    }
    auto cast_op = rewriter.create<BroadCastToOp>(
        loc, target_type, will_be_broadcast,
        llh::buildTensorDims(target_operand, &rewriter), cast_dims);
    if (lhs_type == result_type) {
      rewriter.replaceOpWithNewOp<BinaryOp>(op, result_type, lhs, cast_op);
    } else {
      rewriter.replaceOpWithNewOp<BinaryOp>(op, result_type, cast_op, rhs);
    }
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateInsertBroadCastPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<SimplyBinaryOp<AddOp>>(context);
  patterns.add<SimplyBinaryOp<SubOp>>(context);
  patterns.add<SimplyBinaryOp<DivOp>>(context);
  patterns.add<SimplyBinaryOp<MulOp>>(context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configInsertBroadCastPassConversionTarget(ConversionTarget& target) {
  // target.addIllegalOp<llh::SymbolicBindOp>();
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct InsertBroadCastPass
    : llh::impl::InsertBroadCastPassBase<InsertBroadCastPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void InsertBroadCastPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateInsertBroadCastPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
