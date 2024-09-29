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
#include <cstddef>
#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::llh {
#define COMMON_CHECK                              \
  for (auto res : getOperation()->getResults()) { \
    checkIsReturnOperand(res);                    \
  }
namespace {

void checkIsReturnOperand(Value value) {
  for (auto user : value.getUsers()) {
    if (llvm::isa<mlir::func::ReturnOp>(user)) {
      UNIMPLEMENTED(llc::SymbolInfer);
    }
  }
}
void checkIsIfOperand(Value value) { UNIMPLEMENTED(llc::SymbolInfer); }
void checkIsWhileOperand(Value value) { UNIMPLEMENTED(llc::SymbolInfer); }

void simplyUnarySymbolInfer(Value& value) {
  auto operand_type = value.getDefiningOp()->getOperand(0).getType();
  value.setType(operand_type);
}

void ConvSymbolInfer(Operation* op) {
  auto input_type =
      llvm::cast_or_null<RankedTensorType>(op->getOperand(0).getType());
  CHECK(llc::SymbolInfer, input_type);
  auto weight_type =
      llvm::cast_or_null<RankedTensorType>(op->getOperand(1).getType());
  CHECK(llc::SymbolInfer, weight_type);
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
  auto layout_attr =
      llvm::cast_or_null<LayoutAttr>(op->getAttr(llc::LayoutAttr));
  CHECK(llc::SymbolInfer, layout_attr);
  auto layout = layout_attr.getValue();
  auto dilation_attr =
      llvm::cast_or_null<DenseI64ArrayAttr>(op->getAttr(llc::DilationAttr));
  CHECK(llc::SymbolInfer, dilation_attr);
  auto dilations = dilation_attr.asArrayRef();

  auto rank = input_type.getRank();
  auto new_shape = llvm::SmallVector<int64_t>();
  auto new_shape_symbol = llvm::SmallVector<StringRef>();
  size_t space_index = 0;
  // first compute the space shape
  if (layout == Layout::NCHW) {
    space_index = 2;
  } else if (layout == Layout::NHWC) {
    UNIMPLEMENTED(llc::SymbolInfer);
    space_index = 1;
  }
  auto space_rank = kernel_shape.size();
  auto input_shape = input_type.getShape();
  for (int i = 0; i < space_rank; ++i) {
    auto in = input_shape[i];
    if (!input_type.isDynamicDim(i + space_index)) {
      auto pad_h = pad[i];
      auto pad_l = pad[i + space_rank];
      auto dilation = dilations[i];
      auto stride = strides[i];
      auto kernel_size = kernel_shape[i];
      auto out =
          (in + pad_h + pad_l - dilation * (kernel_size - 1) - 1) / stride + 1;
      new_shape.push_back(out);
      new_shape_symbol.push_back(SymbolAnalysis::UNKOW_SYMBOL);
    } else {
      auto out = ShapedType::kDynamic;
      new_shape.push_back(out);
      new_shape_symbol.push_back(SymbolAnalysis::UNKOW_SYMBOL);
    }
  }
  // batch shape
  auto has_encoding = input_type.getEncoding();
  CHECK(llc::SymbolInfer, has_encoding);
  auto encoding = cast_or_null<EncodingAttr>(has_encoding);
  CHECK(llc::SymbolInfer, encoding);
  auto input_symbols = encoding.getShapeSymbols();
  auto batch_index = 0;
  new_shape.insert(new_shape.begin(), input_shape[batch_index]);
  auto batch_symbol_attr = input_symbols[batch_index];
  auto batch_symbol = batch_symbol_attr.getAttr().str();
  new_shape_symbol.insert(new_shape_symbol.begin(), batch_symbol);
  // channel_shape
  size_t channel_out_index;
  size_t weight_index;
  if (layout == Layout::NCHW) {
    weight_index = 0;
    channel_out_index = 1;
  } else if (layout == Layout::NHWC) {
    UNIMPLEMENTED(llc::SymbolInfer);
    weight_index = 0;
    channel_out_index = rank - 1;
  }
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
  auto symbol_analsis = SymbolAnalysis::getInstance(op);
  auto new_tensor =
      RankedTensorType::get(new_shape, input_type.getElementType());
  op->getResult(0).setType(new_tensor);
  auto res = op->getResult(0);
  symbol_analsis->addEncoding(res, new_shape_symbol);
  UNIMPLEMENTED(llc::SymbolInfer) << "symbol relations";
}
}  // namespace

// checkIsIfOperand(res);                    \
  // checkIsWhileOperand(res);                     \
  }  // namespace mlir::llh
#define UNIMPLEMENTED_INFER_FUNCTION(OP)                                      \
  llvm::LogicalResult OP::inferSymbolicShape() {                              \
    WARN_UNIMPLEMENTED(llc::MLIR) << " op name:" << getOperationName().str(); \
    return llvm::failure();                                                   \
  }

#define INFER_FUNCTION(OP) llvm::LogicalResult OP::inferSymbolicShape()
#define INFER_UNARY_OP(OP)       \
  INFER_FUNCTION(OP) {           \
    auto res = getResult();      \
    simplyUnarySymbolInfer(res); \
    COMMON_CHECK                 \
    return llvm::success();      \
  }
UNIMPLEMENTED_INFER_FUNCTION(MatMulOp)
UNIMPLEMENTED_INFER_FUNCTION(LayerNormOp)
INFER_UNARY_OP(BatchNormOp)
UNIMPLEMENTED_INFER_FUNCTION(DivOp)
UNIMPLEMENTED_INFER_FUNCTION(CatOp)
INFER_UNARY_OP(DropOp)
UNIMPLEMENTED_INFER_FUNCTION(FlattenOp)
UNIMPLEMENTED_INFER_FUNCTION(TransposeOp)
UNIMPLEMENTED_INFER_FUNCTION(AddOp)
UNIMPLEMENTED_INFER_FUNCTION(MulOp)
UNIMPLEMENTED_INFER_FUNCTION(ExpandOp)
UNIMPLEMENTED_INFER_FUNCTION(MaxPoolOp)
UNIMPLEMENTED_INFER_FUNCTION(AdaptiveAvgPoolOp)
INFER_FUNCTION(ConvBiasOp) {
  ConvSymbolInfer(getOperation());
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(ConvOp) {
  ConvSymbolInfer(getOperation());
  COMMON_CHECK
  return llvm::success();
}
INFER_UNARY_OP(ReluOp)
INFER_FUNCTION(WeightOp) {
  auto type = getResult().getType();
  if (!isa<RankedTensorType>(type)) return llvm::failure();
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto res = getResult();
  auto new_res = symbol_analsis->addEncoding(res);
  res.setType(new_res.getType());
  COMMON_CHECK
  return llvm::success();
}
INFER_FUNCTION(ConstantOp) {
  auto type = getResult().getType();
  if (!isa<RankedTensorType>(type)) return llvm::failure();
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto res = getResult();
  symbol_analsis->addEncoding(res);
  COMMON_CHECK
  return llvm::success();
}

INFER_FUNCTION(ReshapeOp) {
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  llvm::SmallVector<int64_t> new_shape;
  auto shapes = getShapes();
  auto res = getResult();
  auto result_type = llvm::cast<RankedTensorType>(res.getType());
  auto symbols = llvm::SmallVector<StringRef>();
  for (auto dim : shapes) {
    if (isConstIntegerValue(dim)) {
      auto val = getConstIntegerValue(dim);
      new_shape.push_back(val);
      auto symbol_op = symbol_analsis->getOrBuildConstSymbolFrom(res, val);
      symbols.push_back(symbol_op.getSymName());
    } else {
      new_shape.push_back(ShapedType::kDynamic);
      auto symbol_op = symbol_analsis->buildNewSymbolFrom(res);
      symbols.push_back(symbol_op.getSymName());
    }
  }
  auto new_res_type =
      RankedTensorType::get(new_shape, result_type.getElementType(),
                            EncodingAttr::get(getContext(), symbols));
  res.setType(new_res_type);
  COMMON_CHECK
  return llvm::success();
}

#undef INFER_FUNCTION
#undef INFER_UNARY_OP
#undef COMMON_CHECK
#undef UNIMPLEMENTED_INFER_FUNCTION
}  // namespace mlir::llh
