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
#include "llcompiler/Dialect/LLH/Utils/InferSymbol.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_INFERSYMBOLSHAPEPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
void generateEntranceSymbol(ModuleOp module) {
  auto funcs = module.getOps<func::FuncOp>();
  auto context = module->getContext();
  auto builder = IRRewriter(context);
  llvm::SmallVector<Type> new_input;
  auto symbol_analysis = SymbolAnalysis::getInstance(module);
  for (auto func : funcs) {
    if (!func->hasAttr(llc::EntranceAttr)) continue;
    auto func_type = func.getFunctionType();
    auto& block = func.getFunctionBody().getBlocks().front();
    auto input_num = block.getNumArguments();
    for (int i{}; i < input_num; i++) {
      auto arg = block.getArgument(i);
      auto new_arg = symbol_analysis->addEncoding(arg);
      new_input.push_back(new_arg.getType());
    }
    auto& blocks = func.getFunctionBody().getBlocks();
    for (auto& block : blocks) {
      if (block.isEntryBlock()) {
        auto new_func_type = FunctionType::get(
            context, new_input, block.getTerminator()->getOperandTypes());
        func.setType(new_func_type);
      }
    }
  }
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateInferSymbolShapePassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct InferSymbolShapePass
    : llh::impl::InferSymbolShapePassBase<InferSymbolShapePass> {
  using InferSymbolShapePassBase::InferSymbolShapePassBase;
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
#include <iostream>
void InferSymbolShapePass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  generateEntranceSymbol(module);
  module.walk([](Operation* op) { checkAndInferSymbol(op); });
  RewritePatternSet patterns(context);
  populateSymbolCanonicalizePatterns(patterns);
  auto analysis = SymbolAnalysis::getInstance(module);
  auto config = GreedyRewriteConfig();
  config.useTopDownTraversal = true;
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
  if (CleanSymbolCache) {
    INFO(llc::SymbolInfer) << "CleanSymbolCache";
    analysis->cleanCache();
  }
  LLC_RUN_OUT_PASS
}
