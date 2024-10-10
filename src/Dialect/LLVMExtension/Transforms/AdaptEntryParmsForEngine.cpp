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

#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::LLVM::ex {
#define GEN_PASS_DEF_ADAPTENTRYPARMSFORENGINEPASS
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h.inc"
}  // namespace mlir::LLVM::ex
using namespace ::mlir;
using namespace ::mlir::LLVM;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct AdaptReturnOp : public OpRewritePattern<LLVM::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult match(LLVM::ReturnOp op) const final {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    CHECK(llc::MLIR_PASS, func);
    if (func->hasAttr(llc::EntranceAttr)) return llvm::success();
    return llvm::failure();
  }
  void rewrite(LLVM::ReturnOp op, PatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    auto operands = op.getOperands();
    auto new_operand = llvm::SmallVector<mlir::Value>();
    for(auto operand : operands){
      auto out_op = operand.getDefiningOp();
      out_op->dump();
      auto types = out_op->getOperandTypes();
      for( auto type:types ){
        type.dump();
      }
    }
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateAdaptEntryParmsForEnginePassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  //patterns.add<AdaptReturnOp>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct AdaptEntryParmsForEnginePass
    : ::LLVM::ex::impl::AdaptEntryParmsForEnginePassBase<
          AdaptEntryParmsForEnginePass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void AdaptEntryParmsForEnginePass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  RewritePatternSet patterns(context);
  populateAdaptEntryParmsForEnginePassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
