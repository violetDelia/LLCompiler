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
#include "llcompiler/Conversion/LLHToHLO/LLHToHLO.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/StablehloOps.h"
namespace mlir {
#define GEN_PASS_DEF_CONVERTLLHTOHLOPASS
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
bool check_const_legal(Operation* op) {
  auto const_op = llvm::cast_or_null<ConstantOp>(op);
  if (!const_op) return false;
  auto type = const_op.getResult().getType();
  return !isa<RankedTensorType>(type);
}

bool check_div_legal(Operation* op) {
  auto div_op = llvm::cast_or_null<DivOp>(op);
  if (!div_op) return false;
  auto type = cast<RankedTensorType>(div_op.getResult().getType());
  return !type.getElementType().isInteger();
}
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct BroadCastToOpToOpLowing : public OpConversionPattern<BroadCastToOp> {
  using OpConversionPattern<BroadCastToOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BroadCastToOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    auto mode = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res);
    auto out_shapes = op.getOutShapes();
    auto cast_dims = op.getCastDims();
    auto operand = op.getInput();
    llvm::SmallVector<Value> out_dims;
    llvm::SmallVector<int64_t> unexpand_dims;
    llvm::SmallVector<int64_t> broadcast_dimensions;
    for (auto shape : out_shapes) {
      auto dim_val =
          rewriter.create<index::CastUOp>(loc, rewriter.getIndexType(), shape);
      out_dims.push_back(dim_val);
    }
    auto rank = res_type.getRank();
    for (int64_t i{}; i < rank; i++) {
      bool is_expand = false;
      for (auto dim : cast_dims) {
        if (dim == i) {
          is_expand = true;
        };
      }
      if (!is_expand) {
        unexpand_dims.push_back(i);
      }
      broadcast_dimensions.push_back(i);
    }
    auto output_dimensions =
        rewriter.create<tensor::FromElementsOp>(loc, out_dims);
    auto new_op = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        loc, res_type, operand, output_dimensions,
        rewriter.getDenseI64ArrayAttr(broadcast_dimensions),
        op.getCastDimsAttr(), rewriter.getDenseI64ArrayAttr(unexpand_dims));
    rewriter.replaceOp(op, new_op);
    return success();
  }
};

struct ConvOpLowing : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;
  // nchw fchw -->nfhw
  // nhwc fhwc -->nhwf
  // curent only supported, need add layout attr and layout pass for more
  LogicalResult matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    auto input = op.getX();
    auto weight = op.getW();
    auto kernal_shape = op.getKernelShape();
    auto kernel_size = kernal_shape.size();
    auto pad = op.getPad();
    auto strides = op.getStride();
    auto dilation = op.getDilation();
    auto graph = op.getGroup();

    auto layout_attr = op.getLayoutAttr();
    auto res = op->getResult(0);
    auto res_type = llc::getRankTensorFrom(res);
    auto spatial_rank = res_type.getRank() - 2;
    auto first_spatial_dim = layout_attr.getFirstSpatialIndex();
    size_t kernel_first_spatial_dim = first_spatial_dim;
    size_t out_first_spatial_dim = first_spatial_dim;
    SmallVector<int64_t> input_spatial_dimensions;
    for (int i = first_spatial_dim; i < first_spatial_dim + spatial_rank; i++) {
      input_spatial_dimensions.push_back(i);
    }
    SmallVector<int64_t> kernel_dimensions;
    for (int64_t i = kernel_first_spatial_dim;
         i < kernel_first_spatial_dim + spatial_rank; i++) {
      kernel_dimensions.push_back(i);
    }
    SmallVector<int64_t> output_spatial_dimensions;
    for (int64_t i = out_first_spatial_dim;
         i < out_first_spatial_dim + spatial_rank; i++) {
      output_spatial_dimensions.push_back(i);
    }

    stablehlo::ConvDimensionNumbersAttr dimension_numbers;
    if (layout_attr.getValue() == Layout::NCHW) {
      dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
          rewriter.getContext(), layout_attr.getBatchIndex(),
          layout_attr.getFeatureIndex(), input_spatial_dimensions, 1, 0,
          kernel_dimensions, layout_attr.getBatchIndex(), 1,
          output_spatial_dimensions);
    } else {
      dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
          rewriter.getContext(), layout_attr.getBatchIndex(),
          layout_attr.getFeatureIndex(), input_spatial_dimensions, 3, 0,
          kernel_dimensions, layout_attr.getBatchIndex(), 3,
          output_spatial_dimensions);
    }

    auto new_stride_attr = rewriter.getDenseI64ArrayAttr(strides);
    auto new_pad_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({spatial_rank, 2}, rewriter.getI64Type()), pad);
    auto new_dilation_attr = rewriter.getDenseI64ArrayAttr(dilation);
    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, res.getType(), input, weight, new_stride_attr, new_pad_attr,
        DenseI64ArrayAttr(), new_dilation_attr, nullptr, dimension_numbers,
        graph, 1, nullptr);
    return success();
  }
};

struct TransposeOpLowing : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern<TransposeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    auto input = op.getInput();
    auto perm = op.getPerms();
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(op, input, perm);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateConvertLLHToHLOPassPatterns(TypeConverter& converter,
                                         RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<SimplyFullLowing<ConstantOp, mlir::stablehlo::ConstantOp>>(
      converter, context);
  patterns.add<SimplyFullLowing<SubOp, stablehlo::SubtractOp>>(converter,
                                                               context);
  patterns.add<SimplyFullLowing<AddOp, stablehlo::AddOp>>(converter, context);
  patterns.add<SimplyFullLowing<MulOp, stablehlo::MulOp>>(converter, context);
  patterns.add<SimplyFullLowing<DivOp, stablehlo::DivOp>>(converter, context);
  patterns.add<SimplyFullLowing<MaxOp, stablehlo::MaxOp>>(converter, context);
  patterns.add<BroadCastToOpToOpLowing>(converter, context);
  patterns.add<ConvOpLowing>(converter, context);
  patterns.add<TransposeOpLowing>(converter, context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configConvertLLHToHLOPassTarget(ConversionTarget& target) {
  target.addDynamicallyLegalOp<ConstantOp>(check_const_legal);
  target.addIllegalOp<DivOp>();
  target.addIllegalOp<SubOp>();
  target.addIllegalOp<AddOp>();
  target.addIllegalOp<MulOp>();
  target.addIllegalOp<BroadCastToOp>();
  target.addIllegalOp<MaxOp>();
  target.addIllegalOp<ReluOp>();
  target.addIllegalOp<ConvOp>();
  target.addIllegalOp<TransposeOp>();
  target.addLegalDialect<stablehlo::StablehloDialect>();
  target.addLegalDialect<mlir::index::IndexDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
}

//===----------------------------------------------------------------------===//
// init typeconvert
//===----------------------------------------------------------------------===//
void initConvertLLHToHLOPassTypeConverter(TypeConverter& converter) {
  auto type_replace = [](Type type) { return type; };
  auto int_replace = [](IntegerType type) { return type; };
  auto index_replace = [](IndexType type) { return type; };
  converter.addConversion(type_replace);
  converter.addConversion(int_replace);
  converter.addConversion(index_replace);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
struct ConvertLLHToHLOPass
    : impl::ConvertLLHToHLOPassBase<ConvertLLHToHLOPass> {
  using impl::ConvertLLHToHLOPassBase<
      ConvertLLHToHLOPass>::ConvertLLHToHLOPassBase;
  void runOnOperation() override;
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//
void ConvertLLHToHLOPass::runOnOperation() {
  LLC_RUN_IN_PASS
  ConversionTarget target(getContext());
  configConvertLLHToHLOPassTarget(target);
  TypeConverter converter;
  initConvertLLHToHLOPassTypeConverter(converter);
  RewritePatternSet patterns(&getContext());
  populateConvertLLHToHLOPassPatterns(converter, patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  RewritePatternSet patterns_special(&getContext());
  LLC_RUN_OUT_PASS
}
