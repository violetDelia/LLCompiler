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
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::llh {

namespace {

void checkIsReturnOperand(Value value) {
  for (auto user : value.getUsers()) {
    if (llvm::isa<mlir::func::ReturnOp>(user)) {
      UNIMPLEMENTED(llc::MLIR);
    }
  }
}
void checkIsIfOperand(Value value) { UNIMPLEMENTED(llc::MLIR); }
void checkIsWhileOperand(Value value) { UNIMPLEMENTED(llc::MLIR); }

void simplyUnarySymbolInfer(Value& value) {
  auto operand_type = value.getDefiningOp()->getOperand(0).getType();
  value.setType(operand_type);
}
}  // namespace
#define COMMON_CHECK                              \
  for (auto res : getOperation()->getResults()) { \
    checkIsIfOperand(res);                        \
    checkIsReturnOperand(res);                    \
    checkIsWhileOperand(res);                     \
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
  COMMON_CHECK
  return llvm::success();
}

#undef INFER_FUNCTION
#undef INFER_UNARY_OP
#undef COMMON_CHECK
#undef UNIMPLEMENTED_INFER_FUNCTION
}  // namespace mlir::llh
