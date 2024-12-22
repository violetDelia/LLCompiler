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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

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
#include "Conversion/LLHToHLO/LLHToHLO.inc"
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
llvm::SmallVector<Value> castToIndex(ConversionPatternRewriter* rewriter,
                                     llvm::SmallVector<Value> values,
                                     Location loc) {
  llvm::SmallVector<Value> new_values;
  for (auto value : values) {
    new_values.push_back(rewriter->create<mlir::arith::IndexCastOp>(
        loc, rewriter->getIndexType(), value));
  }
  return new_values;
}
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
    auto loc = op->getLoc();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res);
    auto out_shapes = op.getOutShapes();
    auto operand = op.getInput();
    auto output_dimensions = rewriter.create<tensor::FromElementsOp>(
        loc, castToIndex(&rewriter, out_shapes, loc));
    auto broadcast_dimensions_attr = op.getCastDimsAttr();
    auto unexpand_dims_attr = op.getNoexpandDimsAttr();
    auto known_expanding_dimensions_attr = op.getExpandDimsAttr();
    auto new_op = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
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
    auto loc = op->getLoc();
    auto input = op.getX();
    auto weight = op.getW();
    auto kernal_shape = op.getKernelShape();
    auto kernel_size = kernal_shape.size();
    auto pad = op.getPad();
    auto stride_attr = op.getStrideAttr();
    auto dilation_attr = op.getDilationAttr();
    auto group = op.getGroup();

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
    } else if (layout_attr.getValue() == Layout::NHWC) {
      dimension_numbers = stablehlo::ConvDimensionNumbersAttr::get(
          rewriter.getContext(), layout_attr.getBatchIndex(),
          layout_attr.getFeatureIndex(), input_spatial_dimensions, 3, 0,
          kernel_dimensions, layout_attr.getBatchIndex(), 3,
          output_spatial_dimensions);
    }
    auto pad_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({spatial_rank, 2}, rewriter.getI64Type()), pad);
    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, res.getType(), input, weight, stride_attr, pad_attr,
        DenseI64ArrayAttr(), dilation_attr, nullptr, dimension_numbers, group,
        1, nullptr);
    return success();
  }
};

struct BatchNormInferenceOpLowing
    : public OpConversionPattern<BatchNormInferenceOp> {
  using OpConversionPattern<BatchNormInferenceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(BatchNormInferenceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
    auto res = op.getResult();
    auto res_type = res.getType();
    auto operand = op.getInput();
    auto scale = op.getScale();
    auto offset = op.getBias();
    auto mean = op.getInputMean();
    auto variance = op.getInputVar();
    auto epsilon = op.getEpsilonAttr();
    auto epsilon_value = epsilon.getValue().convertToDouble();
    auto new_epsilon = rewriter.getF32FloatAttr(epsilon_value);
    auto feature_index = op.getFeatureIndexAttr();
    rewriter.replaceOpWithNewOp<stablehlo::BatchNormInferenceOp>(
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
    auto loc = op->getLoc();
    auto stride = op.getStride();
    auto padding = op.getPadAttr();
    auto kernel_shape = op.getKernelShape();
    auto dilation = op.getDilation();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res);
    auto res_ele_type = res_type.getElementType();
    auto input = op.getInput();
    auto input_type = llc::getRankTensorFrom(input);
    auto input_ele_type = input_type.getElementType();

    auto zore_value = llc::genSplatElementAttr({}, res_ele_type, 0);
    auto init_value = rewriter.create<stablehlo::ConstantOp>(loc, zore_value);

    auto layout = op.getLayoutAttr();
    auto window_dimensions =
        rewriter.getDenseI64ArrayAttr(layout.addBatchAndFeature(kernel_shape));
    auto window_strides =
        rewriter.getDenseI64ArrayAttr(layout.addBatchAndFeature(stride));
    auto window_dilations =
        rewriter.getDenseI64ArrayAttr(layout.addBatchAndFeature(dilation));
    auto base_dilations = DenseI64ArrayAttr();
    auto window_padding = llc::GenWindowPadIntElementsAttr(padding);
    auto reduce_winodw_op = rewriter.create<stablehlo::ReduceWindowOp>(
        loc, res_type, input, init_value, window_dimensions, window_strides,
        base_dilations, window_dilations, window_padding);

    auto& block = reduce_winodw_op.getBody().emplaceBlock();
    auto block_arg1_type = RankedTensorType::get({}, input_ele_type);
    auto block_arg2_type = RankedTensorType::get({}, res_ele_type);
    block.addArgument(block_arg1_type, loc);
    block.addArgument(block_arg2_type, loc);

    rewriter.setInsertionPointToEnd(&block);
    auto max = rewriter.create<stablehlo::MaxOp>(loc, block.getArgument(0),
                                                 block.getArgument(1));
    auto return_op = rewriter.create<stablehlo::ReturnOp>(loc, ValueRange{max});
    rewriter.replaceOp(op, reduce_winodw_op);
  }
};

struct SliceOpLowing : public OpConversionPattern<StrideSliceOp> {
  using OpConversionPattern<StrideSliceOp>::OpConversionPattern;

  LogicalResult match(StrideSliceOp op) const { return llvm::success(); }

  void rewrite(StrideSliceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const {
    auto loc = op->getLoc();
    auto start_index = op.getStartIndex();
    auto end_index = op.getEndIndex();
    auto strides = op.getStrides();
    auto new_satrt_index = rewriter.create<tensor::FromElementsOp>(
        loc, castToIndex(&rewriter, start_index, loc));
    auto new_limits_index = rewriter.create<tensor::FromElementsOp>(
        loc, castToIndex(&rewriter, end_index, loc));
    auto new_strides = rewriter.create<tensor::FromElementsOp>(
        loc, castToIndex(&rewriter, strides, loc));
    rewriter.replaceOpWithNewOp<stablehlo::RealDynamicSliceOp>(
        op, op.getType(), op.getInput(), new_satrt_index, new_limits_index,
        new_strides);
  }
};

struct BatchMatMulOpLowing : public OpConversionPattern<BatchMatMulOp> {
  using OpConversionPattern<BatchMatMulOp>::OpConversionPattern;

  LogicalResult match(BatchMatMulOp op) const { return llvm::success(); }

  void rewrite(BatchMatMulOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    stablehlo::DotDimensionNumbersAttr dotDimensionNumbers =
        stablehlo::DotDimensionNumbersAttr::get(
            rewriter.getContext(),
            /*lhsBatchingDimensions=*/{0},
            /*rhsBatchingDimensions=*/{0},
            /*lhsContractingDimensions=*/{2},
            /*rhsContractingDimensions=*/{1});
    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, op.getType(), lhs, rhs, dotDimensionNumbers, nullptr, nullptr);
  }
};

struct CompareOpLowing : public OpConversionPattern<CompareOp> {
  using OpConversionPattern<CompareOp>::OpConversionPattern;

  LogicalResult match(CompareOp op) const { return llvm::success(); }

  void rewrite(CompareOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const {
    auto context = op->getContext();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto kind_attr = switchCompareAttr(context, op.getKind());
    auto compare_type_attr = genCompareTypeAttr(context, lhs, rhs);
    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, op.getType(), lhs, rhs, kind_attr, compare_type_attr);
  }

  mlir::stablehlo::ComparisonTypeAttr genCompareTypeAttr(
      mlir::MLIRContext* context, Value lhs, Value rhs) const {
    Type lhs_type;
    Type rhs_type;
    if (isa<RankedTensorType>(lhs.getType()))
      lhs_type = llc::getRankTensorFrom(lhs).getElementType();
    else
      lhs_type = lhs.getType();
    if (isa<RankedTensorType>(rhs.getType()))
      rhs_type = llc::getRankTensorFrom(rhs).getElementType();
    else
      rhs_type = rhs.getType();
    CHECK(llc::MLIR_PASS, (lhs_type == rhs_type));
    mlir::stablehlo::ComparisonType conversion_type;
    if (isa<IntegerType>(lhs_type)) {
      auto type = cast<IntegerType>(lhs_type);
      if (type.isUnsigned())
        mlir::stablehlo::ComparisonTypeAttr::get(
            context, stablehlo::ComparisonType::UNSIGNED);
      else
        return mlir::stablehlo::ComparisonTypeAttr::get(
            context, stablehlo::ComparisonType::SIGNED);
    }
    if (isa<FloatType>(lhs_type)) {
      return mlir::stablehlo::ComparisonTypeAttr::get(
          context, stablehlo::ComparisonType::FLOAT);
    }
    UNIMPLEMENTED(llc::MLIR);
  }

  mlir::stablehlo::ComparisonDirectionAttr switchCompareAttr(
      mlir::MLIRContext* context, mlir::llh::CompareKind kind) const {
    if (kind == CompareKind::EQ)
      return mlir::stablehlo::ComparisonDirectionAttr::get(
          context, ::mlir::stablehlo::ComparisonDirection::EQ);
    if (kind == CompareKind::NE)
      return mlir::stablehlo::ComparisonDirectionAttr::get(
          context, ::mlir::stablehlo::ComparisonDirection::NE);
    if (kind == CompareKind::GE)
      return mlir::stablehlo::ComparisonDirectionAttr::get(
          context, ::mlir::stablehlo::ComparisonDirection::GE);
    if (kind == CompareKind::GT)
      return mlir::stablehlo::ComparisonDirectionAttr::get(
          context, ::mlir::stablehlo::ComparisonDirection::GT);
    if (kind == CompareKind::LE)
      return mlir::stablehlo::ComparisonDirectionAttr::get(
          context, ::mlir::stablehlo::ComparisonDirection::LE);
    if (kind == CompareKind::LT)
      return mlir::stablehlo::ComparisonDirectionAttr::get(
          context, ::mlir::stablehlo::ComparisonDirection::LT);
    UNIMPLEMENTED(llc::MLIR);
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateConvertLLHToHLOPassPatterns(TypeConverter& converter,
                                         RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  patterns.add<SimplyFullLowing<ConstantOp, stablehlo::ConstantOp>>(converter,
                                                                    context);
  patterns.add<SimplyFullLowing<SubOp, stablehlo::SubtractOp>>(converter,
                                                               context);
  patterns.add<SimplyFullLowing<AddOp, stablehlo::AddOp>>(converter, context);
  patterns.add<SimplyFullLowing<MulOp, stablehlo::MulOp>>(converter, context);
  patterns.add<SimplyFullLowing<DivOp, stablehlo::DivOp>>(converter, context);
  patterns.add<SimplyFullLowing<MaxOp, stablehlo::MaxOp>>(converter, context);
  patterns.add<SimplyFullLowing<MatMulOp, stablehlo::DotOp>>(converter,
                                                             context);
  patterns.add<SimplyFullLowing<AbsOp, stablehlo::AbsOp>>(converter, context);
  patterns.add<SimplyFullLowing<WhereOp, stablehlo::SelectOp>>(converter,
                                                               context);
  patterns.add<SimplyFullLowing<SqrtOp, stablehlo::SqrtOp>>(converter, context);
  patterns.add<ConvOpLowing>(converter, context);
  patterns.add<TransposeOpLowing>(context);
  patterns.add<BatchNormInferenceOpLowing>(converter, context);
  patterns.add<MaxPoolOpLowing>(converter, context);
  patterns.add<BroadCastToOpToOpLowing>(converter, context);
  patterns.add<SliceOpLowing>(converter, context);
  patterns.add<BatchMatMulOpLowing>(converter, context);
  patterns.add<CompareOpLowing>(converter, context);
}

//===----------------------------------------------------------------------===//
// config target
//===----------------------------------------------------------------------===//
void configConvertLLHToHLOPassTarget(ConversionTarget& target) {
  target.addDynamicallyLegalOp<ConstantOp>(check_const_legal);
  target.addIllegalOp<DivOp, SubOp, AddOp, MulOp, MaxOp, CompareOp, ReluOp,
                      BatchNormOp, AbsOp, SqrtOp, BatchNormInferenceOp, ConvOp,
                      MaxPoolOp, MatMulOp, BatchMatMulOp, TransposeOp,
                      BroadCastToOp, SliceOp, WhereOp>();
  target.addLegalDialect<stablehlo::StablehloDialect>();
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
