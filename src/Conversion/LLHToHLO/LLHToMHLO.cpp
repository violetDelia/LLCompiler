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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Conversion/LLHToHLO/LLHToHLO.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
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
    Loc_And_Context;;
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res);
    auto out_shapes = op.getOutShapes();
    auto cast_dims = op.getCastDims();
    auto operand = op.getInput();
    llvm::SmallVector<Value> out_dims;
    llvm::SmallVector<int64_t> unexpand_dims;
    llvm::SmallVector<int64_t> broadcast_dimensions;
    for (auto shape : out_shapes) {
      auto dim_val = rewriter.create<mlir::arith::IndexCastOp>(
          loc, Index_Ty, shape);
      out_dims.push_back(dim_val);
    }
    auto rank = res_type.getRank();
    for (int64_t i{}; i < rank; i++) {
      bool is_expand = false;
      for (auto dim : cast_dims) {
        if (dim == i) {
          is_expand = true;
        }
      }
      if (!is_expand) {
        unexpand_dims.push_back(i);
      }
      broadcast_dimensions.push_back(i);
    }
    auto output_dimensions =
        rewriter.create<tensor::FromElementsOp>(loc, out_shapes);
    auto i64_type = rewriter.getI64Type();
    auto broadcast_dimensions_shape = RankedTensorType::get(
        {static_cast<int64_t>(broadcast_dimensions.size())}, i64_type);
    auto broadcast_dimensions_attr = DenseIntElementsAttr::get(
        broadcast_dimensions_shape, broadcast_dimensions);
    auto unexpand_dims_shape = RankedTensorType::get(
        {static_cast<int64_t>(unexpand_dims.size())}, i64_type);
    auto unexpand_dims_attr =
        DenseIntElementsAttr::get(unexpand_dims_shape, unexpand_dims);
    auto known_expanding_dimensions_attr =
        llc::ArrayAttrToIntElementsAttr(op.getCastDimsAttr());
    auto new_op = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, res_type, operand, output_dimensions, broadcast_dimensions_attr,
        known_expanding_dimensions_attr, unexpand_dims_attr);
    rewriter.replaceOp(op, new_op);
    return success();
  }
};

struct ConvOpLowing : public OpConversionPattern<ConvOp> {
  using OpConversionPattern<ConvOp>::OpConversionPattern;
  // curent only supported, need add layout attr and layout pass for more
  LogicalResult matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    Loc_And_Context;;
    auto input = op.getX();
    auto weight = op.getW();
    auto kernal_shape = op.getKernelShape();
    auto kernel_size = kernal_shape.size();
    auto pad = op.getPad();
    auto stride_attr = op.getStrideAttr();
    auto dilation_attr = op.getDilationAttr();
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

    mhlo::ConvDimensionNumbersAttr dimension_numbers;
    if (layout_attr.getValue() == Layout::NCHW) {
      dimension_numbers = mhlo::ConvDimensionNumbersAttr::get(
          rewriter.getContext(), layout_attr.getBatchIndex(),
          layout_attr.getFeatureIndex(), input_spatial_dimensions, 1, 0,
          kernel_dimensions, layout_attr.getBatchIndex(), 1,
          output_spatial_dimensions);
    } else {
      dimension_numbers = mhlo::ConvDimensionNumbersAttr::get(
          rewriter.getContext(), layout_attr.getBatchIndex(),
          layout_attr.getFeatureIndex(), input_spatial_dimensions, 3, 0,
          kernel_dimensions, layout_attr.getBatchIndex(), 3,
          output_spatial_dimensions);
    }
    auto new_stride_attr = llc::ArrayAttrToIntElementsAttr(stride_attr);
    auto new_pad_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({spatial_rank, 2}, rewriter.getI64Type()), pad);
    auto new_dilation_attr = llc::ArrayAttrToIntElementsAttr(dilation_attr);
    rewriter.replaceOpWithNewOp<mhlo::ConvolutionOp>(
        op, res.getType(), input, weight, new_stride_attr, new_pad_attr,
        DenseIntElementsAttr(), new_dilation_attr, nullptr, dimension_numbers,
        graph, 1, nullptr);
    return success();
  }
};

struct TransposeOpLowing : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern<TransposeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    auto input = op.getInput();
    auto perm_attr = op.getPermsAttr();
    auto new_perm_attr = llc::ArrayAttrToIntElementsAttr(perm_attr);
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(op, input, new_perm_attr);
    return success();
  }
};

struct BatchNormOpLowing : public OpConversionPattern<BatchNormOp> {
  using OpConversionPattern<BatchNormOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(BatchNormOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    auto res = op.getResult();
    auto res_type = res.getType();
    auto operand = op.getInput();
    auto scale = op.getScale();
    auto offset = op.getBias();
    auto mean = op.getInputMean();
    auto variance = op.getInputVar();
    auto epsilon = op.getEpsilonAttr();
    auto new_epsilon = rewriter.getF32FloatAttr(epsilon.getValueAsDouble());
    auto feature_index = op.getFeatureIndexAttr();
    rewriter.replaceOpWithNewOp<mhlo::BatchNormInferenceOp>(
        op, res_type, operand, scale, offset, mean, variance, new_epsilon,
        feature_index);
    return success();
  }
};

struct MaxPoolOpLowing : public OpConversionPattern<MaxPoolOp> {
  using OpConversionPattern<MaxPoolOp>::OpConversionPattern;

  LogicalResult match(MaxPoolOp op) const { return llvm::success(); }

  void rewrite(MaxPoolOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const {
    Loc_And_Context;;
    auto stride = op.getStrideAttr();
    auto padding = op.getPadAttr();
    auto kernel_shape = op.getKernelShapeAttr();
    auto dilation = op.getDilationAttr();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res);
    auto res_ele_type = res_type.getElementType();
    auto input = op.getInput();
    auto input_type = llc::getRankTensorFrom(input);
    auto input_ele_type = input_type.getElementType();

    auto zore_value = llc::genSplatElementAttr({}, res_ele_type, 0);
    auto init_value = rewriter.create<mhlo::ConstantOp>(loc, zore_value);

    auto layout = op.getLayoutAttr();
    auto window_dimensions =
        llc::GenWindowIntElementsAttr(kernel_shape, layout);
    auto window_strides = llc::GenWindowIntElementsAttr(stride, layout);
    auto window_dilations = llc::GenWindowIntElementsAttr(dilation, layout);
    auto base_dilations = DenseIntElementsAttr();
    auto window_padding = llc::GenWindowPadIntElementsAttr(padding);
    auto reduce_winodw_op = rewriter.create<mhlo::ReduceWindowOp>(
        loc, res_type, input, init_value, window_dimensions, window_strides,
        base_dilations, window_dilations, window_padding);

    auto& block = reduce_winodw_op.getBody().emplaceBlock();
    auto block_arg1_type = RankedTensorType::get({}, input_ele_type);
    auto block_arg2_type = RankedTensorType::get({}, res_ele_type);
    block.addArgument(block_arg1_type, loc);
    block.addArgument(block_arg2_type, loc);

    rewriter.setInsertionPointToEnd(&block);
    auto max = rewriter.create<mhlo::MaxOp>(loc, block.getArgument(0),
                                            block.getArgument(1));
    auto return_op = rewriter.create<mhlo::ReturnOp>(loc, ValueRange{max});
    rewriter.replaceOp(op, reduce_winodw_op);
  }
};

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateConvertLLHToHLOPassPatterns(TypeConverter& converter,
                                         RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<SimplyFullLowing<ConstantOp, mhlo::ConstantOp>>(converter,
                                                               context);
  patterns.add<SimplyFullLowing<SubOp, mhlo::SubtractOp>>(converter, context);
  patterns.add<SimplyFullLowing<AddOp, mhlo::AddOp>>(converter, context);
  patterns.add<SimplyFullLowing<MulOp, mhlo::MulOp>>(converter, context);
  patterns.add<SimplyFullLowing<DivOp, mhlo::DivOp>>(converter, context);
  patterns.add<SimplyFullLowing<MaxOp, mhlo::MaxOp>>(converter, context);
  patterns.add<SimplyFullLowing<MatMulOp, mhlo::DotOp>>(converter, context);
  patterns.add<SimplyFullLowing<AbsOp, mhlo::AbsOp>>(converter, context);
  patterns.add<ConvOpLowing>(converter, context);
  patterns.add<BatchNormOpLowing>(converter, context);
  patterns.add<MaxPoolOpLowing>(converter, context);
  patterns.add<TransposeOpLowing>(converter, context);
  patterns.add<BroadCastToOpToOpLowing>(converter, context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configConvertLLHToHLOPassTarget(ConversionTarget& target) {
  target.addDynamicallyLegalOp<ConstantOp>(check_const_legal);
  target.addIllegalOp<DivOp, SubOp, AddOp, MulOp, MaxOp>();
  target.addIllegalOp<ReluOp, BatchNormOp, AbsOp>();
  target.addIllegalOp<ConvOp, MaxPoolOp>();
  target.addIllegalOp<TransposeOp, BroadCastToOp>();
  target.addLegalDialect<mhlo::MhloDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
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
