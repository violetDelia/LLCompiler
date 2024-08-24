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

#include "llcompiler/Dialect/TosaExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tosa_ex {
#define GEN_PASS_DEF_TRANSFORMLAYOUTTONHWC
#include "llcompiler/Dialect/TosaExtension/Transforms/Passes.h.inc"
}  // namespace mlir::tosa_ex
using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::tosa_ex;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
#define LOG_UNIMPLEMENTED                                             \
  UNIMPLEMENTED(llc::MLIR) << " layout from " << from.str() << " to " \
                           << to.str();

mlir::tosa::ConstOp genTransposeConstOpFromTo(OpBuilder* builder,
                                              StringRef from, StringRef to,
                                              Location& loc) {
  mlir::SmallVector<double> value;
  if (from == llc::layout_to_str(llc::LAYOUT::NCHW)) {
    if (to == llc::layout_to_str(llc::LAYOUT::NHWC)) {
      value.append({0, 2, 3, 1});
    } else {
      UNIMPLEMENTED(llc::MLIR);
    }
  } else if (from == llc::layout_to_str(llc::LAYOUT::NHWC)) {
    if (to == llc::layout_to_str(llc::LAYOUT::NCHW)) {
      value.append({0, 3, 1, 2});
    } else {
      UNIMPLEMENTED(llc::MLIR);
    }
  } else {
    UNIMPLEMENTED(llc::MLIR);
  }
  auto out = RankedTensorType::get({4}, builder->getI64Type());
  auto const_op =
      llc::create_tosa_const(builder, {4}, value, builder->getI64Type(), loc);
  return const_op;
}

mlir::RankedTensorType genReturnTensorFromTo(mlir::Value value, StringRef from,
                                             StringRef to) {
  auto context = value.getContext();
  auto tensor = cast<RankedTensorType>(value.getType());
  CHECK(llc::MLIR, tensor);
  auto shape = tensor.getShape();
  llvm::SmallVector<int64_t> new_shape;
  if (from == llc::layout_to_str(llc::LAYOUT::NCHW)) {
    if (to == llc::layout_to_str(llc::LAYOUT::NHWC)) {
      new_shape.push_back(shape[0]);
      new_shape.push_back(shape[2]);
      new_shape.push_back(shape[3]);
      new_shape.push_back(shape[1]);
    } else {
      UNIMPLEMENTED(llc::MLIR);
    }
  } else if (from == llc::layout_to_str(llc::LAYOUT::NHWC)) {
    if (to == llc::layout_to_str(llc::LAYOUT::NCHW)) {
      new_shape.push_back(shape[0]);
      new_shape.push_back(shape[3]);
      new_shape.push_back(shape[1]);
      new_shape.push_back(shape[2]);
    } else {
      UNIMPLEMENTED(llc::MLIR);
    }
  } else {
    UNIMPLEMENTED(llc::MLIR);
  }
  return RankedTensorType::get(new_shape, tensor.getElementType());
}

bool HaslLayoutAttr(Operation* op, llc::LAYOUT layout) {
  if (!op->hasAttr(llc::LayoutAttr)) return false;
  auto attr = cast<StringAttr>(op->getAttr(llc::LayoutAttr));
  return attr == llc::layout_to_str(layout);
}

mlir::tosa::TransposeOp buildTansposeFromTo(OpBuilder* builder, Value value,
                                            StringRef from, StringRef to,
                                            Location& loc) {
  auto perms = genTransposeConstOpFromTo(builder, from, to, loc);
  auto out = genReturnTensorFromTo(value, from, to);
  auto tranpose = builder->create<tosa::TransposeOp>(loc, out, value, perms);
  return tranpose;
}

DenseI64ArrayAttr genNewPadFrom(DenseI64ArrayAttr pad, StringRef from) {
  auto context = pad.getContext();
  auto pad_value = pad.asArrayRef().vec();
  SmallVector<long long> new_pad_value;
  if (from == llc::layout_to_str(llc::LAYOUT::NCHW)) {
    new_pad_value.push_back(pad_value[0]);
    new_pad_value.push_back(pad_value[2]);
    new_pad_value.push_back(pad_value[3]);
    new_pad_value.push_back(pad_value[1]);
  } else {
    UNIMPLEMENTED(llc::MLIR);
  }
  return DenseI64ArrayAttr::get(context, new_pad_value);
};

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
namespace {
template <class SourceOp, int N>
struct ConvToNHWCWithOperands : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult match(SourceOp op) const final {
    if (!op->hasAttr(llc::LayoutAttr)) return failure();
    return success();
  }

  void rewrite(SourceOp op, PatternRewriter& rewriter) const final {
    LLC_RUN_IN_PATTERN
    auto loc = op->getLoc();
    auto attrs = op->getAttrs();
    auto out = op->getResult(0);
    auto operand_nums = op->getNumOperands();
    auto layout = cast<StringAttr>(op->getAttr(llc::LayoutAttr)).getValue();
    SmallVector<Value> new_operands;
    for (int i = 0; i < N; i++) {
      auto operand = op->getOperand(i);
      auto new_operand =
          buildTansposeFromTo(&rewriter, operand, layout,
                              llc::layout_to_str(llc::LAYOUT::NHWC), loc);
      new_operands.push_back(new_operand);
    }
    for (int i = N; i < operand_nums; i++) {
      new_operands.push_back(op->getOperand(i));
    }
    auto new_out = genReturnTensorFromTo(out, layout,
                                         llc::layout_to_str(llc::LAYOUT::NHWC));
    auto new_op = rewriter.create<SourceOp>(loc, ::mlir::TypeRange{new_out},
                                            new_operands);
    for (auto attr : attrs) {
      if (attr.getName() == llc::LayoutAttr)
        continue;
      else if (attr.getName() == llc::PadAttr) {
        auto pad = cast<DenseI64ArrayAttr>(attr.getValue());
        auto new_pad = genNewPadFrom(pad, layout);
        new_op->setAttr(llc::PadAttr, new_pad);
      } else {
        new_op->setAttr(attr.getName(), attr.getValue());
      }
    }
    auto out_transpose = buildTansposeFromTo(
        &rewriter, new_op, llc::layout_to_str(llc::LAYOUT::NHWC), layout, loc);
    rewriter.replaceOp(op, out_transpose);
    LLC_RUN_OUT_PATTERN
  }
};

}  // namespace
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateTransformLayoutToNHWCPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<ConvToNHWCWithOperands<Conv2DOp, 2>>(context);
  patterns.add<ConvToNHWCWithOperands<MaxPool2dOp, 1>>(context);
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct TransformLayoutToNHWC
    : tosa_ex::impl::TransformLayoutToNHWCBase<TransformLayoutToNHWC> {
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void markOpsNeedTranspose(ModuleOp module) {
  if (!module->hasAttr(llc::GloabalLayoutAttr)) {
    WRONG(llc::MLIR) << "not has " << llc::GloabalLayoutAttr;
    return;
  }
  auto layout = cast<StringAttr>(module->getAttr(llc::GloabalLayoutAttr));
  CHECK(llc::MLIR, layout);
  if (layout == llc::layout_to_str(llc::LAYOUT::NHWC)) return;
  auto mark_op = [layout](Operation* op) {
    if (isa<tosa::Conv2DOp, tosa::MaxPool2dOp>(op)) {
      op->setAttr(llc::LayoutAttr, layout);
      DEBUG(llc::MLIR) << "add " << op->getName().getStringRef().str();
    }
  };
  module->walk(mark_op);
}

void TransformLayoutToNHWC::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  RewritePatternSet patterns(context);
  populateTransformLayoutToNHWCPatterns(patterns);
  auto op = getOperation();
  markOpsNeedTranspose(op);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    signalPassFailure();
  LLC_RUN_OUT_PASS
}

//===----------------------------------------------------------------------===//
// pass create
//===----------------------------------------------------------------------===//
std::unique_ptr<Pass> mlir::tosa_ex::createTransformLayoutToNHWCPass() {
  return std::make_unique<TransformLayoutToNHWC>();
}