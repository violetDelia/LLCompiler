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
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_TRANSFORMLAYOUTPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
namespace {

mlir::DenseI64ArrayAttr getTransposePermsFromLayoutTo(
    mlir::MLIRContext* context, Layout from, Layout to) {
  if (from == Layout::NCHW) {
    if (to == Layout::NCHW) {
      return mlir::DenseI64ArrayAttr::get(context, {{0, 1, 2, 3}});
    } else if (to == Layout::NHWC) {
      return mlir::DenseI64ArrayAttr::get(context, {{0, 2, 3, 1}});
    }
  } else if (from == Layout::NHWC) {
    if (to == Layout::NCHW) {
      return mlir::DenseI64ArrayAttr::get(context, {{0, 3, 1, 2}});
    } else if (to == Layout::NHWC) {
      return mlir::DenseI64ArrayAttr::get(context, {{0, 1, 2, 3}});
    }
  }
  UNIMPLEMENTED(llc::MLIR);
  return {};
}

mlir::RankedTensorType getReturnTensorFromLayoutTo(mlir::Value value,
                                                   Layout from, Layout to) {
  auto context = value.getContext();
  auto tensor = llc::getRankTensorFrom(value);
  auto shape = tensor.getShape();
  llvm::SmallVector<StringRef> symbols;
  bool has_encoding = llc::hasEncoding(tensor);
  if (has_encoding) {
    auto encode = llc::getEncodingFrom(tensor);
    auto encode_symbols = encode.getShapeSymbols();
    for (auto symbol : encode_symbols) {
      symbols.push_back(symbol.getValue());
    }
  }
  llvm::SmallVector<int64_t> new_shape;
  llvm::SmallVector<StringRef, 4> new_symbols;
  if (from == to) {
    for (int i = 0; i < shape.size(); i++) {
      new_shape.push_back(shape[i]);
      if (has_encoding) new_symbols.push_back(symbols[i]);
    }
  } else if (from == Layout::NCHW && to == Layout::NHWC) {
    new_shape.push_back(shape[0]);
    new_shape.push_back(shape[2]);
    new_shape.push_back(shape[3]);
    new_shape.push_back(shape[1]);
    if (has_encoding) {
      new_symbols.push_back(symbols[0]);
      new_symbols.push_back(symbols[2]);
      new_symbols.push_back(symbols[3]);
      new_symbols.push_back(symbols[1]);
    }
  } else if (from == Layout::NHWC && to == Layout::NCHW) {
    new_shape.push_back(shape[0]);
    new_shape.push_back(shape[3]);
    new_shape.push_back(shape[1]);
    new_shape.push_back(shape[2]);
    if (has_encoding) {
      new_symbols.push_back(symbols[0]);
      new_symbols.push_back(symbols[3]);
      new_symbols.push_back(symbols[1]);
      new_symbols.push_back(symbols[2]);
    }
  } else {
    UNIMPLEMENTED(llc::MLIR);
  }
  if (has_encoding) {
    return RankedTensorType::get(new_shape, tensor.getElementType(),
                                 EncodingAttr::get(context, new_symbols));
  }
  return RankedTensorType::get(new_shape, tensor.getElementType());
}

Operation* buildTrnasposeOpFromLayoutTo(LLHPatternRewriter& builder,
                                        Value value, Layout from, Layout to) {
  if (from == to) return value.getDefiningOp();
  auto loc = value.getLoc();
  auto cxt = value.getContext();
  auto tensor = llc::getRankTensorFrom(value);
  auto transpose_perms = getTransposePermsFromLayoutTo(cxt, from, to);
  auto new_type = getReturnTensorFromLayoutTo(value, from, to);
  return builder.create<TransposeOp>(loc, new_type, value, transpose_perms);
}

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct ConvOpConvertLayout : public LLHOpRewritePattern<ConvOp> {
  using LLHOpRewritePattern<ConvOp>::LLHOpRewritePattern;

  explicit ConvOpConvertLayout(MLIRContext* context, Layout target_layout)
      : LLHOpRewritePattern(context), target(target_layout) {}

  LogicalResult match(ConvOp op) const final {
    auto layout = op.getLayout();
    if (layout != target) return llvm::success();
    return llvm::failure();
  }
  void rewrite(ConvOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto context = op->getContext();
    auto maybe_layout = op.getLayout();
    CHECK(llc::MLIR_PASS, maybe_layout.has_value());
    auto layout = maybe_layout.value();
    auto input = op.getX();
    auto weights = op.getW();
    auto res = op->getResult(0);
    auto new_input =
        buildTrnasposeOpFromLayoutTo(rewriter, input, layout, target);
    auto new_weights =
        buildTrnasposeOpFromLayoutTo(rewriter, weights, layout, target);
    auto new_res_type = getReturnTensorFromLayoutTo(res, layout, target);
    auto new_conv = rewriter.create<ConvOp>(
        loc, new_res_type, new_input->getResult(0), new_weights->getResult(0),
        op.getDilation(), op.getKernelShape(), op.getPad(), op.getStride(),
        op.getGroup(), LayoutAttr::get(context, Layout::NHWC),
        LayoutAttr::get(context, Layout::NHWC));
    auto res_transpose =
        buildTrnasposeOpFromLayoutTo(rewriter, new_conv, target, layout);
    rewriter.replaceOp(op, res_transpose);
  }

 protected:
  Layout target;
};

}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
using namespace mlir::llh::impl;
LLC_DEFINE_PASS(
    TransformLayout, { LLC_ADD_PATTERN(ConvOpConvertLayout, TargetLayout); },
    { populateSymbolCanonicalizePatterns(patterns); }, {})
//===----------------------------------------------------------------------===//
// create pass
//===----------------------------------------------------------------------===//
std::unique_ptr<::mlir::Pass> mlir::llh::createTransformLayoutPass() {
  return std::make_unique<TransformLayoutPass>();
}
std::unique_ptr<::mlir::Pass> mlir::llh::createTransformLayoutPass(
    llh::Layout target_layout) {
  ::mlir::llh::TransformLayoutPassOptions options;
  options.TargetLayout = target_layout;
  return std::make_unique<TransformLayoutPass>(options);
}
