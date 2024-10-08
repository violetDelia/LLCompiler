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
#define GEN_PASS_DEF_REMOVEREDUNDANTOPSPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
// discard method
void traverseExpressionSymbolPos(AffineBinaryOpExpr& exp,
                                 SmallVector<size_t>& symbol_pos) {
  auto lhs = exp.getLHS();
  if (auto symbol = llvm::dyn_cast_or_null<AffineSymbolExpr>(lhs)) {
    auto dim_pos = symbol.getPosition();
    exp.shiftSymbols(dim_pos, symbol_pos.size());
    symbol_pos.push_back(dim_pos);
  }
  if (auto dim = llvm::dyn_cast_or_null<AffineDimExpr>(lhs)) {
    auto dim_pos = dim.getPosition();
    symbol_pos.push_back(dim_pos);
  }
  if (auto binary_exp = llvm::dyn_cast_or_null<AffineBinaryOpExpr>(lhs)) {
    traverseExpressionSymbolPos(binary_exp, symbol_pos);
  }
  auto rhs = exp.getRHS();
  if (auto dim = llvm::dyn_cast_or_null<AffineDimExpr>(rhs)) {
    auto dim_pos = dim.getPosition();
    symbol_pos.push_back(dim_pos);
  }
  if (auto binary_exp = llvm::dyn_cast_or_null<AffineBinaryOpExpr>(rhs)) {
    traverseExpressionSymbolPos(binary_exp, symbol_pos);
  }
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct replaceFlattenOp : public LLHOpRewritePattern<FlattenOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(FlattenOp op) const final { return llvm::success(); }
  void rewrite(FlattenOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    auto operand = op->getOperand(0);
    auto result_type = op->getResult(0).getType();
    auto operand_type = operand.getType();
    auto dim_value = op.getDim();
    auto const_dim =
        llvm::dyn_cast_or_null<llh::ConstantOp>(dim_value.getDefiningOp());
    CHECK(llc::MLIR_PASS, const_dim);
    auto dim_attr = llvm::cast_or_null<IntegerAttr>(const_dim.getValueAttr());
    CHECK(llc::MLIR_PASS, dim_attr);
    auto dim = dim_attr.getInt();
    auto dims = buildTensorDims(operand, &rewriter);
    auto reshape_operands = llvm::SmallVector<Value>();
    size_t index = 0;
    for (; index < dim; ++index) {
      auto dim = dims[index];
      reshape_operands.push_back(dim);
    }
    if (index < dims.size()) {
      Value rear_dim_sum = dims[index];
      size_t rear_shape = 1;
      bool is_dynamic = false;
      index++;
      while (index < dims.size()) {
        rear_dim_sum = rewriter.create<MulOp>(loc, rewriter.getI64Type(),
                                              rear_dim_sum, dims[index]);
        index++;
      }
      reshape_operands.push_back(rear_dim_sum);
    }
    auto reshape =
        rewriter.create<ReshapeOp>(loc, result_type, operand, reshape_operands);
    rewriter.replaceOp(op, reshape);
  }
};

std::pair<std::vector<std::string>, mlir::AffineExpr*> generateBindShapeMapKey(
    AffineBinaryOpExpr& exp, SmallVector<size_t>& symbol_pos,
    SymbolicBindOp& op) {
  std::vector<std::string> symbols;
  auto bind_symbols = op.getBindSymbols();
  for (auto pos : symbol_pos) {
    auto symbol = llvm::dyn_cast_or_null<TorchSymbolicIntOp>(
        bind_symbols[pos].getDefiningOp());
    CHECK(llc::MLIR_PASS, symbol);
    symbols.push_back(symbol.getSymName().str());
  }
  return std::make_pair(symbols, &exp);
}

struct replaceTorchSymbolicIntOp
    : public LLHOpRewritePattern<TorchSymbolicIntOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(TorchSymbolicIntOp op) const final {
    if (op->hasAttr(llc::SymbolGeneratedAttr)) return llvm::failure();
    return llvm::success();
  }
  void rewrite(TorchSymbolicIntOp op,
               LLHPatternRewriter& rewriter) const final {
    rewriter.eraseOp(op);
    // auto symbol_analysis = SymbolAnalysis::getInstance();
    // auto symbol = symbol_analysis->buildNewSymbol(&rewriter, op);
    // auto new_name = symbol.getSymNameAttr();
    // op.setSymNameAttr(new_name);
    // llc::add_symbol_generate_attr(op);
  }
};

struct replaceSymbolicBindOp : public LLHOpRewritePattern<SymbolicBindOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(SymbolicBindOp op) const final {
    if (op->hasAttr(llc::StopRunAttr)) return llvm::failure();
    return llvm::success();
  }

  void rewrite(SymbolicBindOp op, LLHPatternRewriter& rewriter) const final {
    rewriter.eraseOp(op);
    // auto operand = op.getOperand();
    // auto bind_shape = op.getBindSymbols();
    // auto bind_type = llvm::cast_or_null<RankedTensorType>(operand.getType());
    // CHECK(llc::MLIR_PASS, bind_type);
    // auto symbol_analysis = SymbolAnalysis::getInstance();
    // auto rank = bind_type.getRank();
    // auto exps_map = op.getExpressions();
    // auto symbol_num = exps_map.getNumSymbols();
    // llvm::SmallVector<StringRef> shapes;
    // for (int i = 0; i < rank; ++i) {
    //   auto exp = exps_map.getResult(i);
    //   if (auto dim = llvm::dyn_cast_or_null<AffineDimExpr>(exp)) {
    //     auto pos = dim.getPosition();
    //     auto symbol_op = llvm::cast_or_null<TorchSymbolicIntOp>(
    //         op.getBindSymbols()[i].getDefiningOp());
    //     CHECK(llc::MLIR_PASS, symbol_op);
    //     auto dim_name = symbol_op.getSymName();
    //     shapes.push_back(dim_name);
    //   } else if (auto const_exp =
    //                  llvm::dyn_cast_or_null<AffineConstantExpr>(exp)) {
    //     auto val = const_exp.getValue();
    //     auto symbol_op =
    //         symbol_analysis->getOrBuildConstSymbol(&rewriter, op, val);
    //     auto dim_name = symbol_op.getSymName();
    //     shapes.push_back(dim_name);
    //   } else if (auto binary_exp =
    //                  llvm::dyn_cast_or_null<AffineBinaryOpExpr>(exp)) {
    //     SmallVector<size_t> symbol_pos;
    //     exps_map.dump();
    //     binary_exp.dump();
    //     traverseExpressionSymbolPos(binary_exp, symbol_pos);
    //     DINFO << "1";
    //     auto key = generateBindShapeMapKey(binary_exp, symbol_pos, op);
    //     if (!bind_shape_map.count(key)) {
    //       DINFO << "1";
    //       llvm::SmallVector<AffineExpr> symReplacements(
    //           symbol_num, rewriter.getAffineSymbolExpr(0));
    //       for (auto sy : symReplacements) {
    //         sy.dump();
    //       }
    //       for (int i = 0; i < symbol_num; i++) {
    //         symReplacements[symbol_pos[i]] = rewriter.getAffineSymbolExpr(i);
    //       }
    //       for (auto sy : symReplacements) {
    //         sy.dump();
    //       }
    //       DINFO << "1";
    //       auto new_exp = binary_exp.replaceDimsAndSymbols({},
    //       symReplacements); new_exp.dump(); auto symbol_op =
    //       symbol_analysis->buildNewSymbol(&rewriter, op); bind_shape_map[key]
    //       = symbol_op.getSymName().str();
    //     }
    //     shapes.push_back(bind_shape_map[key]);
    //   }
    // }
    // llc::add_stop_run_attr(op);
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateRemoveRedundantOpsPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<replaceFlattenOp>(context);
  patterns.add<replaceTorchSymbolicIntOp>(context);
  patterns.add<replaceSymbolicBindOp>(context);
  // patterns.add<replaceSymbolicBindOp>(context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configRemoveRedundantOpsPassConversionTarget(ConversionTarget& target) {
  // target.addIllegalOp<llh::SymbolicBindOp>();
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct RemoveRedundantOpsPass
    : llh::impl::RemoveRedundantOpsPassBase<RemoveRedundantOpsPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void RemoveRedundantOpsPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateRemoveRedundantOpsPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
