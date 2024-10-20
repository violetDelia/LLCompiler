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

#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_UNLOADANDBINDENCODING
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
namespace {
void unloadAndBindEncodingFuncOp(func::FuncOp &func,
                                 LLHPatternRewriter *builder) {
  auto func_type = func.getFunctionType();
  auto input_types = func_type.getInputs();
  auto out_types = func_type.getResults();
  auto &blocks = func.getFunctionBody().getBlocks();
  auto new_out_types = llvm::SmallVector<Type>();
  TypeRange new_input_types;
  for (auto &block : blocks) {
    if (!block.isEntryBlock()) continue;
    auto arg_num = block.getNumArguments();
    for (int i{}; i < arg_num; i++) {
      auto arg = block.getArgument(i);
      auto type = arg.getType();
      mlir::Type new_type = type;
      auto has_encodingg = llc::hasEncoding(type);
      if (has_encodingg) {
        auto tensor = cast<RankedTensorType>(type);
        new_type =
            RankedTensorType::get(tensor.getShape(), tensor.getElementType());
        auto encoding_bind = builder->create<EncodingBindOp>(
            func->getLoc(), ::mlir::TypeRange{}, arg,
            cast<EncodingAttr>(tensor.getEncoding()));
        block.push_front(encoding_bind);
      }
      arg.setType(new_type);
    }
    new_input_types = block.getArgumentTypes();
    auto terminator = block.getTerminator();
    CHECK(llc::MLIR, llvm::isa<func::ReturnOp>(terminator));
    auto return_num = terminator->getNumOperands();
    for (int i{}; i < return_num; i++) {
      auto operand = terminator->getOperand(i);
      auto type = operand.getType();
      if (llc::hasEncoding(type)) {
        auto tensor = cast<RankedTensorType>(type);
        auto new_tensor =
            RankedTensorType::get(tensor.getShape(), tensor.getElementType());
        builder->setInsertionPointAfter(operand.getDefiningOp());
        auto encoding_bind = builder->create<EncodingBindOp>(
            operand.getLoc(), ::mlir::TypeRange{}, operand,
            cast<EncodingAttr>(tensor.getEncoding()));
        operand.setType(new_tensor);
        new_out_types.push_back(new_tensor);
      } else {
        new_out_types.push_back(type);
      }
    }
    auto new_func_type =
        FunctionType::get(func->getContext(), new_input_types, new_out_types);
    func.setFunctionType(new_func_type);
  }
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateUnloadAndBindEncodingPassPatterns(RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  // populateWithGenerated(patterns);
}
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct UnloadAndBindEncodingPass
    : llh::impl::UnloadAndBindEncodingBase<UnloadAndBindEncodingPass> {
  void runOnOperation() override;
};

}  // namespace
void UnloadAndBindEncodingPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto &context = getContext();
  auto module = getOperation();
  auto builder = LLHPatternRewriter(module);
  auto unloda_and_bind_func_attr = [&builder](func::FuncOp func) {
    unloadAndBindEncodingFuncOp(func, &builder);
  };
  auto analysis = SymbolAnalysis::getInstance(module);
  auto unloda_and_bind_encoding = [&analysis, &builder](Operation *op) {
    if (isa<func::FuncOp>(op)) return;
    if (op->getNumResults() == 0) return;
    analysis->buildEncodingBindFrom(op, &builder);
    analysis->unloadEncoding(op);
  };
  auto unloda_and_bind_symbol = [&analysis, &builder](Operation *op) {
    if (isa<func::FuncOp>(op)) return;
    if (op->getNumResults() != 1) return;
    analysis->buildSymbolBindFromAttr(op->getResult(0), &builder);
  };
  module->walk(unloda_and_bind_func_attr);
  module->walk(unloda_and_bind_encoding);
  module->walk(unloda_and_bind_symbol);
  LLC_RUN_OUT_PASS
}
