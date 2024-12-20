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
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/InferSymbol.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
void generateEntranceSymbol(ModuleOp module, bool use_binding = false) {
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
    auto maybe_attrs = func.getArgAttrs();
    if (maybe_attrs.has_value()) {
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
          SmallVector<StringRef> symbols;
          for (size_t dim{}; dim < tensor.getRank(); dim++) {
            auto dim_symbol_attr = arg_attr.getAs<StringAttr>(
                "func.input_symbol_" + std::to_string(dim));
            CHECK(llc::SymbolInfer, dim_symbol_attr);
            symbols.push_back(dim_symbol_attr.getValue());
            symbol_analysis->getOrBuildSymbol(dim_symbol_attr.getValue(), true);
          }
          if (use_binding) {
            symbol_analysis->buildEncodingBindFrom(arg, symbols);
          } else {
            symbol_analysis->addEncoding(arg, symbols);
          }
        }
        new_input.push_back(arg.getType());
      }
    } else {
      for (int i{}; i < input_num; i++) {
        auto arg = block.getArgument(i);
        if (!use_binding) {
          auto new_arg = symbol_analysis->addEncoding(arg);
          new_input.push_back(new_arg.getType());
        } else {
          if (isa<RankedTensorType>(arg.getType())) {
            auto tensor = llc::getRankTensorFrom(arg);
            llvm::SmallVector<StringRef> symbols;
            for (auto dim : tensor.getShape()) {
              if (dim == ShapedType::kDynamic) {
                auto new_symbol = symbol_analysis->buildNewSymbol(true);
                symbols.push_back(new_symbol.getSymName());
              } else {
                auto new_symbol = symbol_analysis->getOrBuildConstSymbol(dim);
                symbols.push_back(new_symbol.getSymName());
              }
            }
            symbol_analysis->buildEncodingBindFrom(arg, symbols);
          }
          new_input.push_back(arg.getType());
        }
      }
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
void InferSymbolShapePass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  auto analysis = SymbolAnalysis::getInstance(module);
  if (UseEncoding) {
    generateEntranceSymbol(module);
  } else {
    generateEntranceSymbol(module, true);
  }
  module.walk([](Operation* op) { checkAndInferSymbol(op); });
  RewritePatternSet patterns(context);
  populateSymbolCanonicalizePatterns(patterns);
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
