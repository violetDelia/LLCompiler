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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::llh {

namespace {
size_t getConstIntegerValue(Value value);
bool isConstIntegerValue(Value value) {
  auto type = value.getType();
  if (!llvm::isa<IntegerType>(type)) return false;
  auto op = value.getDefiningOp();
  if (llvm::isa<DimOp>(op)) {
    auto dim_op = cast<DimOp>(op);
    auto dim_type = llvm::cast<RankedTensorType>(dim_op.getInput().getType());
    CHECK(llc::MLIR, isConstIntegerValue(dim_op.getDim()));
    return !dim_type.isDynamicDim(getConstIntegerValue(dim_op.getDim()));
  }
  if (llvm::isa<ConstantOp>(op)) {
    auto constant_op = llvm::cast<ConstantOp>(op);
    return llvm::isa<IntegerAttr>(constant_op.getValueAttr());
  }
  DINFO << "need fold operator: " << op->getName().getStringRef().str();
  return false;
};

size_t getConstIntegerValue(Value value) {
  auto type = value.getType();
  if (!llvm::isa<IntegerType>(type)) FATAL(llc::MLIR);
  auto op = value.getDefiningOp();
  if (llvm::isa<DimOp>(op)) {
    auto dim_op = cast<DimOp>(op);
    auto dim_type = llvm::cast<RankedTensorType>(dim_op.getInput().getType());
    CHECK(llc::MLIR, isConstIntegerValue(dim_op));
    return dim_type.getDimSize(getConstIntegerValue(dim_op.getDim()));
  }
  if (llvm::isa<ConstantOp>(op)) {
    auto constant_op = llvm::cast<ConstantOp>(op);
    if (!llvm::isa<IntegerAttr>(constant_op.getValueAttr())) FATAL(llc::MLIR);
    return llvm::cast<IntegerAttr>(constant_op.getValueAttr()).getInt();
  }
  UNIMPLEMENTED(llc::MLIR);
}

void simplyUnarySymbolInfer(Value& value) {
  auto operand_type = value.getDefiningOp()->getOperand(0).getType();
  value.setType(operand_type);
}
}  // namespace

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
    return llvm::success();      \
  }
UNIMPLEMENTED_INFER_FUNCTION(MatMulOp)
UNIMPLEMENTED_INFER_FUNCTION(LayerNormOp)
UNIMPLEMENTED_INFER_FUNCTION(BatchNormOp)
UNIMPLEMENTED_INFER_FUNCTION(DivOp)
UNIMPLEMENTED_INFER_FUNCTION(CatOp)
INFER_UNARY_OP(DropOp)
UNIMPLEMENTED_INFER_FUNCTION(FlattenOp)
UNIMPLEMENTED_INFER_FUNCTION(TransposeOp)
UNIMPLEMENTED_INFER_FUNCTION(AddOp)
UNIMPLEMENTED_INFER_FUNCTION(ConvBiasOp)
UNIMPLEMENTED_INFER_FUNCTION(MulOp)
UNIMPLEMENTED_INFER_FUNCTION(ExpandOp)
UNIMPLEMENTED_INFER_FUNCTION(MaxPoolOp)
UNIMPLEMENTED_INFER_FUNCTION(AdaptiveAvgPoolOp)
UNIMPLEMENTED_INFER_FUNCTION(ConvOp)
INFER_UNARY_OP(ReluOp)
INFER_FUNCTION(WeightOp) {
  auto type = getResult().getType();
  if (!isa<RankedTensorType>(type)) return llvm::success();
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto res = getResult();
  auto new_res = symbol_analsis->addEncoding(res);
  res.setType(new_res.getType());
  return llvm::success();
}
INFER_FUNCTION(ConstantOp) {
  auto type = getResult().getType();
  if (!isa<RankedTensorType>(type)) return llvm::failure();
  auto symbol_analsis = SymbolAnalysis::getInstance(getOperation());
  auto res = getResult();
  symbol_analsis->addEncoding(res);
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
      symbol_op->dump();
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
  return llvm::success();
}

#undef INFER_FUNCTION
#undef INFER_UNARY_OP
#undef UNIMPLEMENTED_INFER_FUNCTION
}  // namespace mlir::llh
