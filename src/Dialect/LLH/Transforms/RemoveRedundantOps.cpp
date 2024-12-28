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

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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

void generateEntranceTensorEncoding(ModuleOp module) {
  auto funcs = module.getOps<func::FuncOp>();
  auto context = module->getContext();
  auto builder = IRRewriter(context);
  llvm::SmallVector<Type> new_input;
  for (auto func : funcs) {
    if (!func->hasAttr(llc::EntranceAttr)) continue;
    auto func_type = func.getFunctionType();
    auto& block = func.getFunctionBody().getBlocks().front();
    auto input_num = block.getNumArguments();
    auto maybe_attrs = func.getArgAttrs();
    if (!maybe_attrs.has_value()) return;
    auto attrs = maybe_attrs.value().getValue();
    CHECK(llc::MLIR_PASS,
          attrs.size() == func.getFunctionType().getNumInputs());
    for (int i{}; i < input_num; i++) {
      auto arg = block.getArgument(i);
      if (isa<RankedTensorType>(arg.getType())) {
        auto tensor = llvm::cast<RankedTensorType>(arg.getType());
        auto has_encode = tensor.getEncoding();
        auto arg_attr = llvm::cast<DictionaryAttr>(attrs[i]);
        CHECK(llc::SymbolInfer, arg_attr);
        auto symbols_system = SymbolAnalysis::getInstance(module);
        SmallVector<StringRef> symbols;
        for (size_t dim{}; dim < tensor.getRank(); dim++) {
          auto dim_symbol_attr = arg_attr.getAs<StringAttr>(
              "func.input_symbol_" + std::to_string(dim));
          CHECK(llc::SymbolInfer, dim_symbol_attr);
          symbols.push_back(dim_symbol_attr.getValue());
          symbols_system->getOrBuildSymbol(dim_symbol_attr.getValue(), true);
        }
        symbols_system->addEncoding(arg, symbols);
      }
      new_input.push_back(arg.getType());
    }
    auto& blocks = func.getFunctionBody().getBlocks();
    for (auto& sub_block : blocks) {
      if (sub_block.isEntryBlock()) {
        auto new_func_type = FunctionType::get(
            context, new_input, sub_block.getTerminator()->getOperandTypes());
        func.setType(new_func_type);
      }
    }
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
      auto input_dim = dims[index];
      reshape_operands.push_back(input_dim);
    }
    if (index < dims.size()) {
      Value rear_dim_sum = dims[index];
      index++;
      while (index < dims.size()) {
        rear_dim_sum = rewriter.create<MulOp>(
            loc, rewriter.getI64Type(), ValueRange{rear_dim_sum, dims[index]});
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
    AffineBinaryOpExpr& exp, const SmallVector<size_t>& symbol_pos,
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
    return llvm::success();
  }
  void rewrite(TorchSymbolicIntOp op,
               LLHPatternRewriter& rewriter) const final {
    auto func = op->getParentOfType<func::FuncOp>();
    CHECK(llc::MLIR, func);
    auto loc = op->getLoc();
    auto symbol = op.getSymName();
    auto maybe_attrs = func.getArgAttrs();
    auto& blocks = func.getBlocks();
    if (maybe_attrs.has_value()) {
      auto func_type = func.getFunctionType();
      auto& block = func.getFunctionBody().getBlocks().front();
      auto input_num = block.getNumArguments();
      auto attrs = maybe_attrs.value().getValue();
      for (int i{}; i < input_num; i++) {
        auto arg = block.getArgument(i);
        if (!isa<RankedTensorType>(arg.getType())) continue;
        auto tensor = llvm::cast<RankedTensorType>(arg.getType());
        auto arg_attr = llvm::cast<DictionaryAttr>(attrs[i]);
        for (size_t dim{}; dim < tensor.getRank(); dim++) {
          auto dim_symbol_attr = arg_attr.getAs<StringAttr>(
              "func.input_symbol_" + std::to_string(dim));
          if (!dim_symbol_attr) continue;
          if (dim_symbol_attr.getValue() == symbol) {
            auto dim_op = llh::buildTensorDim(arg, &rewriter, dim);
            rewriter.replaceOp(op, dim_op);
            return;
          }
        }
      }
    }
    for (auto& block : blocks) {
      if (!block.isEntryBlock()) continue;
      for (auto arg : block.getArguments()) {
        auto type = arg.getType();
        if (!isa<RankedTensorType>(type)) continue;
        if (llc::hasEncoding(type)) {
          auto encoding = llc::getEncodingFrom(type);
          auto symbols = encoding.getShapeSymbols();
          for (size_t i{}; i < symbols.size(); i++) {
            auto name = symbols[i].getValue();
            if (name != symbol) continue;
            auto dim_op = llh::buildTensorDim(arg, &rewriter, i);
            rewriter.replaceOp(op, dim_op);
            return;
          }
        }
      }
    }
    WARN(llc::MLIR) << "not find symbol dim!";
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
  //generateEntranceTensorEncoding(module);
  RewritePatternSet patterns(context);
  populateRemoveRedundantOpsPassPatterns(patterns);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
