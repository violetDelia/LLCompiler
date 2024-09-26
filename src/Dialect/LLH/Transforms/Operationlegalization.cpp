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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Transforms/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_OPERATIONLEGALIZATION
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
struct BraodcastableScalarToTensor : public LLCOpRewritePattern<ConstantOp> {
  using LLCOpRewritePattern::LLCOpRewritePattern;
  LogicalResult match(ConstantOp op) const final {
    if (op.use_empty()) return llvm::failure();
    if (!op->getResult(0).getType().isIntOrFloat()) return llvm::failure();
    for (auto user : op->getUsers()) {
      if (user->hasTrait<OpTrait::ResultsBroadcastableShape>()) {
        return llvm::success();
      }
    }
    return llvm::failure();
  }
  void rewrite(ConstantOp op, LLCPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    auto type = op->getResult(0).getType();
    auto tensor_type = RankedTensorType::get({1}, type);
    auto const_tensor = rewriter.create<ConstantOp>(
        loc, tensor_type,
        DenseElementsAttr::get(tensor_type, op.getValueAttr()));
    for (auto user : op->getUsers()) {
      if (user->hasTrait<OpTrait::ResultsBroadcastableShape>()) {
        auto operand_num = user->getNumOperands();
        for (int i = 0; i < operand_num; i++) {
          auto operand = user->getOperand(i);
          if (operand.getDefiningOp() == op) {
            user->setOperand(i, const_tensor);
          }
        }
      }
    }
  }
};


//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateOperationlegalizatioPassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<BraodcastableScalarToTensor>(context);
  //patterns.add<replaceTorchSymbolicIntOp>(context);
  //patterns.add<replaceSymbolicBindOp>(context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configOperationlegalizatioConversionTarget(ConversionTarget& target) {
  // target.addIllegalOp<llh::SymbolicBindOp>();
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct OperationlegalizatioPass
    : llh::impl::OperationlegalizationBase<OperationlegalizatioPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void OperationlegalizatioPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  RewritePatternSet patterns(context);
  populateOperationlegalizatioPassPatterns(patterns);
  auto op = getOperation();
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}

//===----------------------------------------------------------------------===//
// pass create
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> mlir::llh::createOperationlegalizationPass() {
  return std::make_unique<OperationlegalizatioPass>();
}
