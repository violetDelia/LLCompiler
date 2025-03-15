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
#include "llcompiler/Conversion/StablehlotoLinalgExtension/StablehlotoLinalgExtension.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llcompiler/Support/MlirUtility.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOLINALGEXTENSIONPASS
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// legal func
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
// delete check for bound
struct HLORealDynamicSliceOpToLinalg final
    : OpConversionPattern<mlir::stablehlo::RealDynamicSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  static Value computeSize(Location loc, Value start, Value limit, Value stride,
                           ConversionPatternRewriter& rewriter) {
    Value ret = Arith_CeilDivUI(Arith_SubI(limit, start), stride);
    if (ret.getType().isIndex()) return ret;
    return IndexCast(rewriter.getIndexType(), ret);
  }

  LogicalResult match(mlir::stablehlo::RealDynamicSliceOp op) const final {
    auto start_indexs = op.getODSOperands(1).begin();
    auto element_type = getElementTypeOrSelf(*start_indexs);
    if (getElementTypeOrSelf(*op.getODSOperands(2).begin()) != element_type ||
        getElementTypeOrSelf(*op.getODSOperands(3).begin()) != element_type) {
      FATAL(llc::MLIR_PASS) << "dimension element type is not same";
      return llvm::failure();
    }
    return llvm::success();
  }

  void rewrite(mlir::stablehlo::RealDynamicSliceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const override {
    Loc_And_Context;
    auto input = adaptor.getOperand();
    auto input_type = llc::getRankTensorFrom(input);
    auto start_indexs = adaptor.getStartIndices();
    auto element_type = getElementTypeOrSelf(start_indexs);
    auto arith_type = element_type.isIndex() ? I64_Ty : element_type;
    auto res_type = llvm::cast<RankedTensorType>(
        this->typeConverter->convertType(op.getType()));
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<Type, 3> clamp_type(3, arith_type);
    for (auto i : llvm::seq<unsigned>(0, input_type.getRank())) {
      Value dim = ConstantIndex(i);
      Value start = Tensor_Extract(adaptor.getStartIndices(), dim);
      Value limit = Tensor_Extract(adaptor.getLimitIndices(), dim);
      Value stride = Tensor_Extract(adaptor.getStrides(), dim);
      auto res_dim = res_type.getDimSize(i);
      Value size = ShapedType::isDynamic(res_dim)
                       ? computeSize(loc, start, limit, stride, rewriter)
                       : ConstantIndex(res_dim);
      if (!start.getType().isIndex()) start = IndexCast(Index_Ty, start);
      offsets.push_back(start);
      if (ShapedType::isDynamic(res_dim))
        sizes.push_back(size);
      else
        sizes.push_back(Index_Attr(res_dim));
      if (!stride.getType().isIndex()) stride = IndexCast(Index_Ty, stride);
      strides.push_back(stride);
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, res_type, input, offsets, sizes, strides);
  }
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
LLC_DEFINE_CONVERSION_PASS(
    ConvertStablehloToLinalgExtension,
    {LLC_ADD_CONVERSION(HLORealDynamicSliceOpToLinalg)},
    {
      target.addIllegalOp<stablehlo::RealDynamicSliceOp>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<tensor::TensorDialect>();
    },
    {
      auto index_repalce = [](IndexType type) { return type; };
      auto int_repalce = [](IntegerType type) { return type; };
      auto shaped_repalce = [](ShapedType type) { return type; };
      auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
      converter.addConversion(ranked_tensor_replace);
      converter.addConversion(shaped_repalce);
      converter.addConversion(int_repalce);
      converter.addConversion(index_repalce);
    })
