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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_TRANSFORMLAYOUTTONHWC
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
namespace {

// mlir::DenseI64ArrayAttr genTransposePermsToNHWC(mlir::Value value,
//                                                 llc::LAYOUT src) {
//   auto context = value.getContext();
//   if (src == llc::LAYOUT::NCHW) {
//     return mlir::DenseI64ArrayAttr::get(context, {{0, 2, 3, 1}});
//   }
//   if (src == llc::LAYOUT::NHWC) {
//     return mlir::DenseI64ArrayAttr::get(context, {{0, 1, 2, 3}});
//   }
//   UNIMPLEMENTED(llc::MLIR);
//   return {};
// }

// mlir::DenseI64ArrayAttr genTransposePermsFromNHWC(mlir::Value value,
//                                                   llc::LAYOUT target) {
//   auto context = value.getContext();
//   if (target == llc::LAYOUT::NHWC) {
//     return mlir::DenseI64ArrayAttr::get(context, {{0, 3, 1, 2}});
//   }
//   if (target == llc::LAYOUT::NCHW) {
//     return mlir::DenseI64ArrayAttr::get(context, {{0, 1, 2, 3}});
//   }
//   UNIMPLEMENTED(llc::MLIR);
//   return {};
// }

// mlir::RankedTensorType genReturnTensorFrom(mlir::Value value, llc::LAYOUT src) {
//   auto context = value.getContext();
//   auto tensor = cast<RankedTensorType>(value.getType());
//   auto shape = tensor.getShape();
//   llvm::SmallVector<int64_t> new_shape;
//   if (src == llc::LAYOUT::NHWC) {
//     for (auto val : shape) {
//       new_shape.push_back(val);
//     }
//   }
//   if (src = llc::LAYOUT::NCHW) {
//     new_shape.push_back(shape[0]);
//     new_shape.push_back(shape[2]);
//     new_shape.push_back(shape[3]);
//     new_shape.push_back(shape[1]);
//   } else {
//     UNIMPLEMENTED(llc::MLIR);
//   }
//   return RankedTensorType::get(new_shape, tensor.getElementType());
// }

// bool HaslLayoutAttr(mlir::Value value, llc::LAYOUT layout) {
//   auto op = value.getDefiningOp();
//   if (!op->hasAttr(llc::LayoutAttr)) return false;
//   auto attr = cast<StringAttr>(op->getAttr(llc::LayoutAttr));
//   return attr == llc::layout_to_str(layout);
// }
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
// #include "llcompiler/Dialect/LLH/Transforms/TransformLayoutToNHWC.inc"
// struct ConvOpToNHWC : public OpRewritePattern<ConvOp> {
//   using OpRewritePattern::OpRewritePattern;
//   void rewrite(ConvOp op, PatternRewriter& rewriter) const final { ; }
//   LogicalResult match(ConvOp op) const final {
//     llvm_unreachable("must override match or matchAndRewrite");
//   }
//};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateTransformLayoutToNHWCPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  //populateWithGenerated(patterns);
}
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct TransformLayoutToNHWC
    : llh::impl::TransformLayoutToNHWCBase<TransformLayoutToNHWC> {
  void runOnOperation() override;
  void markOpsNeedTranspose(ModuleOp module);
};

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void TransformLayoutToNHWC::markOpsNeedTranspose(ModuleOp module) {
  auto layout = cast<StringAttr>(module->getAttr(llc::GloabalLayoutAttr));
  CHECK(llc::MLIR, layout);
  // if (layout == llc::layout_to_str(llc::LAYOUT::NHWC)) return;
  // auto mark_op = [layout](Operation* op) {
  //   if (isa<llh::ConvOp>(op)) {
  //     op->setAttr(llc::LayoutAttr, layout);
  //   }
  // };
  // module->walk(mark_op);
}
}  // namespace
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
std::unique_ptr<Pass> mlir::llh::createTransformLayoutToNHWCPass() {
  return std::make_unique<TransformLayoutToNHWC>();
}
