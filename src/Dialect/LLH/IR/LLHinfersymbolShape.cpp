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
#include <iterator>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "symengine/add.h"
#include "symengine/constants.h"
#include "symengine/integer.h"
#include "symengine/mul.h"
#include "symengine/printers.h"
#include "symengine/simplify.h"

namespace mlir::llh {
using SymEngine::add;
using SymEngine::div;
using SymEngine::integer;
using SymEngine::mul;
using SymEngine::one;
using SymEngine::sub;
using SymEngine::zero;

#define COMMON_CHECK                              \
  for (auto res : getOperation()->getResults()) { \
    checkIsReturnOperand(res);                    \
  }
namespace {

int getChannelOutIndex(Layout layout, int rank) {
  switch (layout) {
    case Layout::NCHW:
      return 1;
    case Layout::NHWC:
      return rank - 1;
  }
  UNIMPLEMENTED(llc::SymbolInfer);
  return -1;
}

void checkIsReturnOperand(Value& value) {
  for (auto user : value.getUsers()) {
    if (llvm::isa<mlir::func::ReturnOp>(user)) {
      auto func = user->getParentOfType<func::FuncOp>();
      auto func_type = func.getFunctionType();
      auto new_func_type = FunctionType::get(
          value.getContext(), func_type.getInputs(), user->getOperandTypes());
      func.setFunctionType(new_func_type);
    }
  }
}
// 遍历shapes ()
bool encodingWithDims(mlir::Operation* op, OperandRange dims,
                      ShapedType input_type) {
  auto symbol_analsis = SymbolAnalysis::getInstance(op);
  auto symbols = llvm::SmallVector<StringRef>();
  auto new_shapes = llvm::SmallVector<int64_t>();
  for (auto dim : dims) {
    auto symbol = symbol_analsis->getOrBuildSymbolAttrFrom(dim);
    symbols.push_back(symbol);
    if (llh::isConstIntegerValue(dim)) {
      new_shapes.push_back(llh::getConstIntegerValue(dim));
    } else {
      new_shapes.push_back(ShapedType::kDynamic);
    }
  }
  auto new_tensor =
      RankedTensorType::get(new_shapes, input_type.getElementType());
  op->getResult(0).setType(new_tensor);
  auto res = op->getResult(0);
  symbol_analsis->addEncoding(res, symbols);
  return true;
}

void simplyUnarySymbolInfer(Value& value) {
  auto operand_type = value.getDefiningOp()->getOperand(0).getType();
  value.setType(operand_type);
}

void simplyBinarySymbolInfer(Value& value) {
  auto op = value.getDefiningOp();
  auto input1 = op->getOperand(0);
  auto input2 = op->getOperand(1);
  auto input1_type = dyn_cast<RankedTensorType>(input1.getType());
  auto input2_type = dyn_cast<RankedTensorType>(input2.getType());
  if (!input1_type || !input2_type) {
    WRONG(llc::SymbolInfer) << "UnrankTensor!";
    return;
  }
  if (input1_type.getRank() != input2_type.getRank()) {
    RankedTensorType base_type;
    if (input1_type.getRank() > input2_type.getRank()) {
      base_type = input1_type;
    } else {
      base_type = input2_type;
    }
    value.setType(base_type);
    return;
  }
  auto symbol_analsis = SymbolAnalysis::getInstance(op);
  auto rank = input1_type.getRank();
  auto new_shape = llvm::SmallVector<int64_t>();
  auto new_shape_symbol = llvm::SmallVector<StringRef>();
  for (size_t i = 0; i < rank; i++) {
    auto encoding1 = dyn_cast<EncodingAttr>(input1_type.getEncoding());
    auto encoding2 = dyn_cast<EncodingAttr>(input2_type.getEncoding());
    if (!encoding1 || !encoding2) {
      WRONG(llc::SymbolInfer) << "Unmatched Encoding!";
      return;
    }
    auto dim1 = input1_type.getDimSize(i);
    auto dim2 = input2_type.getDimSize(i);
    if (!input1_type.isDynamicDim(i) && !input2_type.isDynamicDim(i)) {
      new_shape.push_back(std::max({dim1, dim2}));
      new_shape_symbol.push_back(SymbolAnalysis::UNKOW_SYMBOL);
      continue;
    }
    auto symbol_1 = encoding1.getShapeSymbols()[i].getValue();
    auto symbol_2 = encoding2.getShapeSymbols()[i].getValue();
    if (dim1 == 1) {
      new_shape.push_back(dim2);
      new_shape_symbol.push_back(symbol_2);
      continue;
    }
    if (dim2 == 1) {
      new_shape.push_back(dim1);
      new_shape_symbol.push_back(symbol_1);
      continue;
    }
    new_shape.push_back(ShapedType::kDynamic);
    if (symbol_1.str() == symbol_2.str()) {
      new_shape_symbol.push_back(symbol_1);
    } else {
      new_shape_symbol.push_back(symbol_1);
      symbol_analsis->buildSymbolRelation(symbol_1, symbol_2,
                                          SymbolRelation::EQ);
    }
  }
  auto new_tensor =
      RankedTensorType::get(new_shape, input1_type.getElementType());
  op->getResult(0).setType(new_tensor);
  auto res = op->getResult(0);
  symbol_analsis->addEncoding(res, new_shape_symbol);
}

void ConvSymbolInfer(Operation* op) {
  // first compute the space shape
  auto symbol_analsis = SymbolAnalysis::getInstance(op);
  auto context = op->getContext();
  size_t space_index = 0;
  auto layout_attr =
      llvm::cast_or_null<LayoutAttr>(op->getAttr(llc::LayoutAttr));
  CHECK(llc::SymbolInfer, layout_attr);
  auto layout = layout_attr.getValue();
  if (layout == Layout::NCHW) {
    space_index = 2;
  } else if (layout == Layout::NHWC) {
    UNIMPLEMENTED(llc::SymbolInfer);
    space_index = 1;
  }
  auto input_type =
      llvm::cast_or_null<RankedTensorType>(op->getOperand(0).getType());
  CHECK(llc::SymbolInfer, input_type);
  auto kernel_shape_attr =
      llvm::cast_or_null<DenseI64ArrayAttr>(op->getAttr(llc::KernelShapeAttr));
  CHECK(llc::SymbolInfer, kernel_shape_attr);
  auto kernel_shape = kernel_shape_attr.asArrayRef();
  auto pad_attr =
      llvm::cast_or_null<DenseI64ArrayAttr>(op->getAttr(llc::PadAttr));
  CHECK(llc::SymbolInfer, pad_attr);
  auto pad = pad_attr.asArrayRef();
  auto stride_attr =
      llvm::cast_or_null<DenseI64ArrayAttr>(op->getAttr(llc::StrideAtt));
  auto strides = stride_attr.asArrayRef();
  auto dilation_attr =
      llvm::cast_or_null<DenseI64ArrayAttr>(op->getAttr(llc::DilationAttr));
  CHECK(llc::SymbolInfer, dilation_attr);
  auto dilations = dilation_attr.asArrayRef();
  auto rank = input_type.getRank();
  auto new_shape = llvm::SmallVector<int64_t>();
  auto new_shape_symbol = llvm::SmallVector<StringRef>();
  auto space_rank = kernel_shape.size();
  auto input_shape = input_type.getShape();
  auto has_encoding = input_type.getEncoding();
  CHECK(llc::SymbolInfer, has_encoding);
  auto encoding = cast_or_null<EncodingAttr>(has_encoding);
  CHECK(llc::SymbolInfer, encoding);
  auto input_symbols = encoding.getShapeSymbols();
  for (int i = 0; i < space_rank; ++i) {
    auto in = input_shape[i + space_rank];
    auto pad_h = pad[i];
    auto pad_l = pad[i + space_rank];
    auto dilation = dilations[i];
    auto stride = strides[i];
    auto kernel_size = kernel_shape[i];
    if (!input_type.isDynamicDim(i + space_index)) {
      auto out =
          (in + pad_h + pad_l - dilation * (kernel_size - 1) - 1) / stride + 1;
      new_shape.push_back(out);
      new_shape_symbol.push_back(SymbolAnalysis::UNKOW_SYMBOL);
    } else {
      auto out = ShapedType::kDynamic;
      new_shape.push_back(out);
      auto in_basic_symbol = symbol_analsis->getBasicSymbol(
          input_symbols[i + space_rank].getValue());
      (in + pad_h + pad_l - dilation * (kernel_size - 1) - 1) / stride + 1;

      auto new_symbol = add(
          div(add(in_basic_symbol,
                  integer(pad_h + pad_l - dilation * (kernel_size - 1) - 1)),
              integer(stride)),
          one);
      auto exp =
          getAffineBinaryOpExpr(AffineExprKind::CeilDiv,
                                (getAffineSymbolExpr(0, context) + pad_h +
                                 pad_l - dilation * (kernel_size - 1) - 1),
                                getAffineConstantExpr(stride, context)) +
          1;
      auto affine_map = AffineMap::get(1, 1, exp);
      auto symbol_op = symbol_analsis->buildNewSymbol(
          new_symbol, affine_map, {input_symbols[i + space_rank].getValue()},
          true);
      new_shape_symbol.push_back(symbol_op.getSymName());
    }
  }
  // batch shape
  auto batch_index = 0;
  new_shape.insert(new_shape.begin(), input_shape[batch_index]);
  auto batch_symbol_attr = input_symbols[batch_index];
  auto batch_symbol = batch_symbol_attr.getAttr().str();
  new_shape_symbol.insert(new_shape_symbol.begin(), batch_symbol);
  // channel_shape
  size_t channel_out_index{};
  size_t weight_index{};
  if (layout == Layout::NCHW) {
    weight_index = 0;
    channel_out_index = 1;
  } else if (layout == Layout::NHWC) {
    UNIMPLEMENTED(llc::SymbolInfer);
    weight_index = 0;
    channel_out_index = rank - 1;
  }
  auto weight_type =
      llvm::cast_or_null<RankedTensorType>(op->getOperand(1).getType());
  CHECK(llc::SymbolInfer, weight_type);
  auto weight_has_encoding = weight_type.getEncoding();
  auto weight_shape = weight_type.getShape();
  CHECK(llc::SymbolInfer, weight_has_encoding);
  auto weight_encoding = cast_or_null<EncodingAttr>(weight_has_encoding);
  CHECK(llc::SymbolInfer, weight_encoding);
  auto weight_symbols = weight_encoding.getShapeSymbols();
  new_shape.insert(new_shape.begin() + channel_out_index,
                   weight_shape[weight_index]);
  auto channel_out_symbol_attr = weight_symbols[weight_index];
  auto channel_out_symbol = channel_out_symbol_attr.getAttr().str();
  new_shape_symbol.insert(new_shape_symbol.begin() + channel_out_index,
                          channel_out_symbol);
  auto new_tensor =
      RankedTensorType::get(new_shape, input_type.getElementType());
  op->getResult(0).setType(new_tensor);
  auto res = op->getResult(0);
  symbol_analsis->addEncoding(res, new_shape_symbol);
}
}  // namespace

#define HAS_ENCODING_RETURN(value) \
  if (llc::hasEncoding(value)) return llvm::failure();

#define HAS_SYMBOLATTR_RETURN(op) \
  if (op->hasAttr(llc::SymbolIntAttr)) return llvm::failure();

#define NO_SYMBOLATTR_RETURN(value)                        \
  if (!value.getDefiningOp()->hasAttr(llc::SymbolIntAttr)) \
    return llvm::failure();

#define NO_ENCODING_RETURN(value) \
  if (!llc::hasEncoding(value)) return llvm::failure();

#define UNIMPLEMENTED_INFER_FUNCTION(OP)                                      \
  llvm::LogicalResult OP::inferSymbolicShape() {                              \
    WARN_UNIMPLEMENTED(llc::MLIR) << " op name:" << getOperationName().str(); \
    return llvm::failure();                                                   \
  }

#define INFER_FUNCTION(OP) llvm::LogicalResult OP::inferSymbolicShape()

// 一元op
#define INFER_UNARY_OP(OP)                            \
  INFER_FUNCTION(OP) {                                \
    NO_ENCODING_RETURN(getOperation()->getOperand(0)) \
    HAS_ENCODING_RETURN(getOperation()->getResult(0)) \
    auto res = getResult();                           \
    simplyUnarySymbolInfer(res);                      \
    COMMON_CHECK                                      \
    return llvm::success();                           \
  }
// binary op
#define INFER_BINARY(OP)                                                 \
  INFER_FUNCTION(OP) {                                                   \
    HAS_ENCODING_RETURN(getOperation()->getResult(0))                    \
    HAS_SYMBOLATTR_RETURN(getOperation())                                \
    if (isa<IntegerType, IntegerType>(getOperand(0).getType())) {        \
      auto symbol_analsis = SymbolAnalysis::getInstance(getOperation()); \
      symbol_analsis->getOrBuildSymbolAttrFrom(getOperation());          \
      return llvm::success();                                            \
    }                                                                    \
    NO_ENCODING_RETURN(getOperation()->getOperand(0))                    \
    NO_ENCODING_RETURN(getOperation()->getOperand(1))                    \
    auto res = getResult();                                              \
    simplyBinarySymbolInfer(res);                                        \
    COMMON_CHECK                                                         \
    return llvm::success();                                              \
  }

// conv类op
#define INFER_CONV(OP)                                \
  INFER_FUNCTION(OP) {                                \
    NO_ENCODING_RETURN(getOperation()->getOperand(0)) \
    NO_ENCODING_RETURN(getOperation()->getOperand(1)) \
    HAS_ENCODING_RETURN(getOperation()->getResult(0)) \
    ConvSymbolInfer(getOperation());                  \
    COMMON_CHECK                                      \
    return llvm::success();                           \
  }
// 没有操作数的op
#define INFER_NO_OPERAND(OP)                                            \
  INFER_FUNCTION(OP) {                                                  \
    auto symbol_analysis = SymbolAnalysis::getInstance(getOperation()); \
    for (auto res : getOperation()->getResults()) {                     \
      if (!isa<RankedTensorType>(res.getType())) continue;              \
      auto new_res = symbol_analysis->addEncoding(res);                 \
      res.setType(new_res.getType());                                   \
    }                                                                   \
    COMMON_CHECK                                                        \
    return llvm::success();                                             \
  }

INFER_UNARY_OP(DropOp)
INFER_UNARY_OP(ReluOp)
INFER_UNARY_OP(AbsOp)
INFER_UNARY_OP(ConvertToOp)
INFER_UNARY_OP(SqrtOp)
INFER_UNARY_OP(RsqrtOp)
INFER_UNARY_OP(BatchNormInferenceOp)

INFER_BINARY(AddOp)
INFER_BINARY(MulOp)
INFER_BINARY(DivOp)
INFER_BINARY(SubOp)
INFER_BINARY(MaxOp)
INFER_BINARY(MinOp)
INFER_BINARY(CompareOp)

INFER_CONV(ConvBiasOp)
INFER_CONV(ConvOp)
INFER_NO_OPERAND(WeightOp)
INFER_FUNCTION(ConstantOp) {
  auto res = getResult();
  HAS_ENCODING_RETURN(res)
  HAS_SYMBOLATTR_RETURN(getOperation())
  auto value = getValueAttr();
  if (isa<IntegerAttr>(value)) {
    auto symbol_analysis = SymbolAnalysis::getInstance(getOperation());
    symbol_analysis->getOrBuildSymbolAttrFrom(res);
    COMMON_CHECK
    return llvm::success();
  }
  if (isa<DenseElementsAttr>(value)) {
    auto symbol_analysis = SymbolAnalysis::getInstance(getOperation());
    auto dense_attr = llvm::cast<DenseElementsAttr>(value);
    auto shape = dense_attr.getType();
    auto new_res_type =
        RankedTensorType::get(shape.getShape(), shape.getElementType());
    res.setType(new_res_type);
    symbol_analysis->addEncoding(getResult());
    COMMON_CHECK return llvm::success();
  }
  return llvm::failure();
}

UNIMPLEMENTED_INFER_FUNCTION(LayerNormOp)
UNIMPLEMENTED_INFER_FUNCTION(CatOp)
UNIMPLEMENTED_INFER_FUNCTION(FlattenOp)
UNIMPLEMENTED_INFER_FUNCTION(ExpandOp)
UNIMPLEMENTED_INFER_FUNCTION(SliceOp)
INFER_FUNCTION(StrideSliceOp) {
  HAS_ENCODING_RETURN(getResult())
  NO_ENCODING_RETURN(getInput())
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto symbols = llvm::SmallVector<StringRef>();
  auto new_shapes = llvm::SmallVector<int64_t>();
  auto input_type = llc::getRankTensorFrom(getInput());
  auto input_symbols = llc::getEncodingFrom(input_type).getShapeSymbols();
  auto start_indexs = getStartIndex();
  auto end_indexs = getEndIndex();
  auto strides = getStrides();
  for (auto [input_symbol, start, end, stride] :
       llvm::zip(input_symbols, start_indexs, end_indexs, strides)) {
    auto start_symbol = symbol_analsis->getOrBuildSymbolAttrFrom(start);
    auto end_symbol = symbol_analsis->getOrBuildSymbolAttrFrom(end);
    auto stride_symbol = symbol_analsis->getOrBuildSymbolAttrFrom(stride);
    auto tem_symbol = symbol_analsis->buildNewSymbolWithRelation(
        end_symbol, start_symbol, SymbolRelation::Sub);
    auto res_symbol = symbol_analsis->buildNewSymbolWithRelation(
        tem_symbol.getSymName(), stride_symbol, SymbolRelation::FloorDiv);
    new_shapes.push_back(symbol_analsis->getIntValue(res_symbol.getSymName()));
    symbols.push_back(res_symbol.getSymName());
  }
  auto new_tensor =
      RankedTensorType::get(new_shapes, input_type.getElementType());
  getResult().setType(new_tensor);
  auto res = getResult();
  symbol_analsis->addEncoding(res, symbols);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(ExtractOp) {
  HAS_ENCODING_RETURN(getResult())
  NO_ENCODING_RETURN(getInput())
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto symbols = llvm::SmallVector<StringRef>();
  auto new_shapes = llvm::SmallVector<int64_t>();
  auto input_type = llc::getRankTensorFrom(getInput());
  auto input_symbols = llc::getEncodingFrom(input_type).getShapeSymbols();
  auto rank = input_type.getRank();
  if (rank == 1) {
    new_shapes.push_back(1);
    symbols.push_back(SymbolAnalysis::UNKOW_SYMBOL);
  } else {
    for (int i = 1; i < rank; ++i) {
      new_shapes.push_back(input_type.getShape()[i]);
      symbols.push_back(input_symbols[i].getValue());
    }
  }
  auto new_tensor =
      RankedTensorType::get(new_shapes, input_type.getElementType());
  getResult().setType(new_tensor);
  auto res = getResult();
  symbol_analsis->addEncoding(res, symbols);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(MatMulOp) {
  HAS_ENCODING_RETURN(getResult())
  NO_ENCODING_RETURN(getLhs())
  NO_ENCODING_RETURN(getRhs())
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto symbols = llvm::SmallVector<StringRef>();
  auto new_shapes = llvm::SmallVector<int64_t>();
  auto lhs_type = llc::getRankTensorFrom(getLhs());
  auto rhs_type = llc::getRankTensorFrom(getRhs());
  auto lhs_symbols = llc::getEncodingFrom(lhs_type).getShapeSymbols();
  auto rhs_symbols = llc::getEncodingFrom(rhs_type).getShapeSymbols();
  new_shapes.push_back(lhs_type.getShape()[0]);
  new_shapes.push_back(rhs_type.getShape()[1]);
  symbols.push_back(lhs_symbols[0].getValue());
  symbols.push_back(rhs_symbols[1].getValue());
  auto new_tensor =
      RankedTensorType::get(new_shapes, lhs_type.getElementType());
  getResult().setType(new_tensor);
  auto res = getResult();
  symbol_analsis->addEncoding(res, symbols);
  COMMON_CHECK
  symbol_analsis->buildSymbolRelation(lhs_symbols[1].getAttr().strref(),
                                      rhs_symbols[0].getAttr().strref(),
                                      SymbolRelation::EQ);
  return llvm::success();
}

INFER_FUNCTION(BatchMatMulOp) {
  HAS_ENCODING_RETURN(getResult())
  NO_ENCODING_RETURN(getLhs())
  NO_ENCODING_RETURN(getRhs())
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto symbols = llvm::SmallVector<StringRef>();
  auto new_shapes = llvm::SmallVector<int64_t>();
  auto lhs_type = llc::getRankTensorFrom(getLhs());
  auto rhs_type = llc::getRankTensorFrom(getRhs());
  auto lhs_symbols = llc::getEncodingFrom(lhs_type).getShapeSymbols();
  auto rhs_symbols = llc::getEncodingFrom(rhs_type).getShapeSymbols();
  new_shapes.push_back(lhs_type.getShape()[0]);
  new_shapes.push_back(lhs_type.getShape()[1]);
  new_shapes.push_back(rhs_type.getShape()[2]);
  symbols.push_back(lhs_symbols[0].getValue());
  symbols.push_back(lhs_symbols[1].getValue());
  symbols.push_back(rhs_symbols[2].getValue());
  auto new_tensor =
      RankedTensorType::get(new_shapes, lhs_type.getElementType());
  getResult().setType(new_tensor);
  auto res = getResult();
  symbol_analsis->addEncoding(res, symbols);
  COMMON_CHECK
  symbol_analsis->buildSymbolRelation(lhs_symbols[2].getAttr().strref(),
                                      rhs_symbols[1].getAttr().strref(),
                                      SymbolRelation::EQ);
  return llvm::success();
}

INFER_FUNCTION(AdaptiveAvgPoolOp) {
  NO_ENCODING_RETURN(getOperation()->getOperand(0))
  HAS_ENCODING_RETURN(getOperation()->getResult(0))
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto out_size = getOutSize();
  auto symbols = llvm::SmallVector<StringRef>();
  auto new_shapes = llvm::SmallVector<int64_t>();
  auto input_type = llc::getRankTensorFrom(getInput());
  auto input_symbols = llc::getEncodingFrom(input_type).getShapeSymbols();
  auto out_size_len = out_size.size();
  auto remind_len = input_type.getRank() - out_size_len;
  for (int i = 0; i < remind_len; i++) {
    new_shapes.push_back(input_type.getDimSize(i));
    symbols.push_back(input_symbols[i].getValue());
  }
  for (int i = 0; i < out_size_len; i++) {
    new_shapes.push_back(out_size[i]);
    symbols.push_back(SymbolAnalysis::UNKOW_SYMBOL);
  }
  auto new_tensor =
      RankedTensorType::get(new_shapes, input_type.getElementType());
  getResult().setType(new_tensor);
  auto res = getResult();
  symbol_analsis->addEncoding(res, symbols);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(MaxPoolOp) {
  NO_ENCODING_RETURN(getOperation()->getOperand(0))
  HAS_ENCODING_RETURN(getOperation()->getResult(0))
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto context = getContext();
  auto layout_attr =
      llvm::cast_or_null<LayoutAttr>(getOperation()->getAttr(llc::LayoutAttr));
  CHECK(llc::SymbolInfer, layout_attr);
  auto layout = layout_attr.getValue();
  size_t space_index = layout_attr.getFirstSpatialIndex();
  auto input_type = llc::getRankTensorFrom(getInput());
  auto kernel_shape = getKernelShape();
  auto pad = getPad();
  auto strides = getStride();
  auto dilations = getDilation();
  auto rank = input_type.getRank();
  auto new_shape = llvm::SmallVector<int64_t>();
  auto new_shape_symbol = llvm::SmallVector<StringRef>();
  auto space_rank = kernel_shape.size();
  auto input_shape = input_type.getShape();
  auto has_encoding = input_type.getEncoding();
  CHECK(llc::SymbolInfer, has_encoding);
  auto encoding = cast_or_null<EncodingAttr>(has_encoding);
  CHECK(llc::SymbolInfer, encoding);
  auto input_symbols = encoding.getShapeSymbols();
  for (int i = 0; i < space_rank; ++i) {
    auto in = input_shape[i + space_rank];
    auto pad_h = pad[i];
    auto pad_l = pad[i + space_rank];
    auto dilation = dilations[i];
    auto stride = strides[i];
    auto kernel_size = kernel_shape[i];
    if (!input_type.isDynamicDim(i + space_index)) {
      auto out =
          (in + pad_h + pad_l - dilation * (kernel_size - 1) - 1) / stride + 1;
      new_shape.push_back(out);
      new_shape_symbol.push_back(SymbolAnalysis::UNKOW_SYMBOL);
    } else {
      auto out = ShapedType::kDynamic;
      new_shape.push_back(out);
      auto in_basic_symbol = symbol_analsis->getBasicSymbol(
          input_symbols[i + space_rank].getValue());
      auto new_symbol = div(
          sub(add(add(in_basic_symbol, integer(pad_h)), integer(pad_l)),
              sub(mul(integer(dilation), sub(integer(kernel_size), one)), one)),
          integer(stride));
      auto exp =
          getAffineBinaryOpExpr(AffineExprKind::CeilDiv,
                                (getAffineSymbolExpr(0, context) + pad_h +
                                 pad_l - dilation * (kernel_size - 1) - 1),
                                getAffineConstantExpr(stride, context)) +
          1;
      auto affine_map = AffineMap::get(1, 1, exp);
      auto symbol_op = symbol_analsis->buildNewSymbol(
          new_symbol, affine_map, {input_symbols[i + space_rank].getValue()},
          true);
      new_shape_symbol.push_back(symbol_op.getSymName());
    }
  }
  // batch shape
  auto batch_index = 0;
  new_shape.insert(new_shape.begin(), input_shape[batch_index]);
  auto batch_symbol_attr = input_symbols[batch_index];
  auto batch_symbol = batch_symbol_attr.getAttr().str();
  new_shape_symbol.insert(new_shape_symbol.begin(), batch_symbol);
  // channel_shape
  size_t channel_index = getChannelOutIndex(layout, rank);
  new_shape.insert(new_shape.begin() + channel_index,
                   input_shape[channel_index]);
  auto channel_out_symbol_attr = input_symbols[channel_index];
  auto channel_out_symbol = channel_out_symbol_attr.getAttr().str();
  new_shape_symbol.insert(new_shape_symbol.begin() + channel_index,
                          channel_out_symbol);
  auto new_tensor =
      RankedTensorType::get(new_shape, input_type.getElementType());
  getResult().setType(new_tensor);
  auto res = getResult();
  symbol_analsis->addEncoding(res, new_shape_symbol);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(BroadCastToOp) {
  NO_ENCODING_RETURN(getInput())
  HAS_ENCODING_RETURN(getResult())
  auto dims = getOutShapes();
  auto input_type = llc::getRankTensorFrom(getInput());
  if (!encodingWithDims(getOperation(), dims, input_type)) {
    WARN_UNIMPLEMENTED(llc::SymbolInfer);
    return llvm::failure();
  }
  auto input_symbols = llc::getEncodingFrom(getInput()).getShapeSymbols();
  auto res_symbols = llc::getEncodingFrom(getResult()).getShapeSymbols();
  auto cast_dims = getCastDims();
  auto input_rank = input_symbols.size();
  auto expand_dims = llvm::SmallVector<int64_t>();
  auto unexpand_dims = llvm::SmallVector<int64_t>();
  for (int i = 0; i < input_rank; i++) {
    auto in_symbol = input_symbols[i];
    auto out_symbol = res_symbols[cast_dims[i]];
    if (in_symbol == out_symbol)
      unexpand_dims.push_back(i);
    else
      expand_dims.push_back(i);
  }
  setExpandDims(expand_dims);
  setNoexpandDims(unexpand_dims);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(ReshapeOp) {
  NO_ENCODING_RETURN(getInput())
  HAS_ENCODING_RETURN(getResult())
  auto dims = getShapes();
  auto input_type = llc::getRankTensorFrom(getInput());
  if (!encodingWithDims(getOperation(), dims, input_type)) {
    WARN_UNIMPLEMENTED(llc::SymbolInfer);
    return llvm::failure();
  }
  COMMON_CHECK
  INFO_UNIMPLEMENTED(llc::SymbolInfer) << "symbol relations";
  return llvm::success();
}

INFER_FUNCTION(EmptyOp) {
  HAS_ENCODING_RETURN(getResult())
  auto dims = getShapes();
  auto input_type = llc::getRankTensorFrom(getResult());
  if (!encodingWithDims(getOperation(), dims, input_type)) {
    WARN_UNIMPLEMENTED(llc::SymbolInfer);
    return llvm::failure();
  }
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(TransposeOp) {
  NO_ENCODING_RETURN(getInput())
  HAS_ENCODING_RETURN(getResult())
  auto prem = getPerms();
  auto input_type = getInput().getType();
  CHECK(llc::SymbolInfer, llvm::isa<RankedTensorType>(input_type));
  auto input_tensor = llvm::cast<RankedTensorType>(input_type);
  auto encoding_attr = input_tensor.getEncoding();
  CHECK(llc::SymbolInfer, llvm::isa<EncodingAttr>(encoding_attr));
  auto encoding = cast<EncodingAttr>(encoding_attr);
  auto input_symbols = encoding.getShapeSymbols();
  auto shape = input_tensor.getShape();
  llvm::SmallVector<int64_t> new_shape;
  auto new_symbols = llvm::SmallVector<StringRef>();
  for (size_t i = 0; i < prem.size(); ++i) {
    auto prem_val = prem[i];
    new_shape.push_back(shape[prem_val]);
    new_symbols.push_back(input_symbols[prem_val].getValue());
  }
  auto new_res_type =
      RankedTensorType::get(new_shape, input_tensor.getElementType(),
                            EncodingAttr::get(getContext(), new_symbols));
  getResult().setType(new_res_type);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(BatchNormOp) {
  NO_ENCODING_RETURN(getInput())
  HAS_ENCODING_RETURN(getResult())
  auto res = getResult();
  res.setType(getInput().getType());
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  symbol_analsis->addEncoding(getRunningMean());
  symbol_analsis->addEncoding(getRunningVar());
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(WhereOp) {
  NO_ENCODING_RETURN(getPred())
  NO_ENCODING_RETURN(getOnTrue())
  NO_ENCODING_RETURN(getOnFalse())
  HAS_ENCODING_RETURN(getResult())
  auto res = getResult();
  res.setType(getOnTrue().getType());
  COMMON_CHECK
  return llvm::success();
}

#undef INFER_FUNCTION
#undef INFER_UNARY_OP
#undef INFER_BINARY
#undef INFER_NO_OPERAND

#undef INFER_CONV
#undef COMMON_CHECK
#undef UNIMPLEMENTED_INFER_FUNCTION
#undef HAS_ENCODING_RETURN
#undef HAS_SYMBOLATTR_RETURN
#undef NO_SYMBOLATTR_RETURN
#undef NO_ENCODING_RETURN
}  // namespace mlir::llh
