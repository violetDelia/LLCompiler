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
#define GEN_PASS_DEF_OPERATIONLEGALIZATIONPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

// op type only int/float
ConstantOp buildConstTensorFromScalar(ConstantOp op,
                                      LLHPatternRewriter* rewriter,
                                      Operation* user) {
  auto user_type =
      llvm::dyn_cast_or_null<ShapedType>(user->getResult(0).getType());
  CHECK(llc::MLIR_PASS, user_type);
  auto user_ele_type = user_type.getElementType();
  auto tensor_type = RankedTensorType::get({1}, user_ele_type);
  auto const_type = op->getResult(0).getType();
  auto loc = op->getLoc();
  DenseElementsAttr new_value;
  if (user_ele_type.isInteger()) {
    if (const_type.isInteger()) {
      auto data_attr = llvm::dyn_cast_or_null<IntegerAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      new_value = DenseElementsAttr::get(tensor_type, {data});
    } else {
      auto data_attr = llvm::dyn_cast_or_null<FloatAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      auto int_attr = IntegerAttr::get(user_ele_type, data.convertToFloat());
      new_value = DenseElementsAttr::get(tensor_type, {int_attr.getValue()});
    }
  } else {
    if (const_type.isInteger()) {
      auto data_attr = llvm::dyn_cast_or_null<IntegerAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      auto float_attr = FloatAttr::get(user_ele_type, data.bitsToDouble());
      new_value = DenseElementsAttr::get(tensor_type, {float_attr.getValue()});
    } else {
      auto data_attr = llvm::dyn_cast_or_null<FloatAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      auto float_attr = FloatAttr::get(user_ele_type, data.convertToFloat());
      new_value = DenseElementsAttr::get(tensor_type, {float_attr.getValue()});
    }
  }
  return rewriter->create<ConstantOp>(loc, tensor_type, new_value);
}

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct BraodcastableScalarToTensor : public LLHOpRewritePattern<ConstantOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
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
  void rewrite(ConstantOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    for (auto user : op->getUsers()) {
      if (user->hasTrait<OpTrait::ResultsBroadcastableShape>()) {
        auto operand_num = user->getNumOperands();
        auto const_tensor = buildConstTensorFromScalar(op, &rewriter, user);
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
  // patterns.add<replaceSymbolicBindOp>(context);
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
    : llh::impl::OperationlegalizationPassBase<OperationlegalizatioPass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void OperationlegalizatioPass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  auto module = getOperation();
  // mark layout
  auto global_layout = module->getAttr(llc::GloabalLayoutAttr);
  CHECK(llc::MLIR_PASS, llvm::isa<StringAttr>(global_layout));
  auto layout = symbolizeLayout(dyn_cast<StringAttr>(global_layout).getValue());
  CHECK(llc::MLIR_PASS, layout.has_value());
  auto add_layout_attr = [&layout](Operation* op) {
    if (!isLayoutSensitive(op)) return;
    if (!op->hasAttr(llc::LayoutAttr)) {
      llc::add_layout_attr(op, layout.value());
    }
  };
  module->walk(add_layout_attr);
  RewritePatternSet patterns(context);
  populateOperationlegalizatioPassPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}
