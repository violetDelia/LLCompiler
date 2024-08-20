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

#include "llcompiler/Dialect/IRExtension/IR/Attrs.h"
#include "llcompiler/Dialect/IRExtension/IR/Enums.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_TRANSFORMLAYOUTTONHWC
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace mlir;
using namespace mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
mlir::DenseI64ArrayAttr genTransposePermsToNHWC(mlir::Value value) {
  auto context = value.getContext();
  auto tensor = cast<RankedTensorType>(value.getType());
  auto encode = cast<mlir::ex::EncodingAttr>(tensor.getEncoding());
  if (encode.getLayout() == mlir::ex::Layout::NCHW) {
    return mlir::DenseI64ArrayAttr::get(context, {{0, 2, 3, 1}});
  }
  if (encode.getLayout() == mlir::ex::Layout::NHWC) {
    return mlir::DenseI64ArrayAttr::get(context, {{0, 1, 2, 3}});
  }
  UNIMPLEMENTED(llc::MLIR);
  return {};
}

mlir::DenseI64ArrayAttr genTransposePermsFromNHWC(mlir::Value value) {
  auto context = value.getContext();
  auto tensor = cast<RankedTensorType>(value.getType());
  auto encode = cast<mlir::ex::EncodingAttr>(tensor.getEncoding());
  if (encode.getLayout() == mlir::ex::Layout::NCHW) {
    return mlir::DenseI64ArrayAttr::get(context, {{0, 3, 1, 2}});
  }
  if (encode.getLayout() == mlir::ex::Layout::NHWC) {
    return mlir::DenseI64ArrayAttr::get(context, {{0, 1, 2, 3}});
  }
  UNIMPLEMENTED(llc::MLIR);
  return {};
}

mlir::RankedTensorType genNHWCReturnTensor(mlir::Value value) {
  auto context = value.getContext();
  auto tensor = cast<RankedTensorType>(value.getType());
  auto layout = cast<mlir::ex::EncodingAttr>(tensor.getEncoding()).getLayout();
  auto encode = mlir::ex::EncodingAttr::get(context, mlir::ex::Layout::NHWC);
  auto shape = tensor.getShape();
  llvm::SmallVector<int64_t> new_shape;
  if (layout == mlir::ex::Layout::NHWC) {
    for (auto val : shape) {
      new_shape.push_back(val);
    }
  } else if (layout == mlir::ex::Layout::NCHW) {
    new_shape.push_back(shape[0]);
    new_shape.push_back(shape[2]);
    new_shape.push_back(shape[3]);
    new_shape.push_back(shape[1]);
  } else {
    UNIMPLEMENTED(llc::MLIR);
  }
  return RankedTensorType::get(new_shape, tensor.getElementType(), encode);
}
mlir::RankedTensorType genNHWCReturnTensor(mlir::TypeRange types) {
  auto type = types[0];
  auto context = type.getContext();
  auto tensor = cast<RankedTensorType>(type);
  auto encode = mlir::ex::EncodingAttr::get(context, mlir::ex::Layout::NHWC);
  return RankedTensorType::get(tensor.getShape(), tensor.getElementType(),
                               encode);
}

bool layoutIsNotNHWC(mlir::Value value) {
  auto tensor = cast<mlir::RankedTensorType>(value.getType());
  CHECK(llc::MLIR, tensor);
  return llc::getLayoutFrom(tensor) != mlir::ex::Layout::NHWC;
}

bool layoutIsSame(mlir::Value left, Value right) {
  auto left_tensor = cast<mlir::RankedTensorType>(left.getType());
  CHECK(llc::MLIR, left_tensor);
  auto right_tensor = cast<mlir::RankedTensorType>(right.getType());
  CHECK(llc::MLIR, right_tensor);
  return llc::getLayoutFrom(right_tensor) == llc::getLayoutFrom(left_tensor);
}

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
namespace {
#include "llcompiler/Dialect/LLH/Transforms/TransformLayoutToNHWC.inc"
}
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateTransformLayoutToNHWCPatterns(RewritePatternSet& patterns) {
  populateWithGenerated(patterns);
}
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
namespace {
struct TransformLayoutToNHWC
    : llh::impl::TransformLayoutToNHWCBase<TransformLayoutToNHWC> {
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void TransformLayoutToNHWC::runOnOperation() {
  LLC_RUN_IN_PASS
  auto* context = &getContext();
  RewritePatternSet patterns(context);
  populateTransformLayoutToNHWCPatterns(patterns);
  auto op = getOperation();
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