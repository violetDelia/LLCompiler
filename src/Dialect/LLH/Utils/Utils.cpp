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

#include "llcompiler/Dialect/LLH/Utils/Utils.h"

#include <cstddef>

namespace mlir::llh {

llh::DimOp buildTensorDim(mlir::Value operand, RewriterBase* rewrite,
                          size_t dim) {
  auto loc = operand.getLoc();
  auto dim_const = rewrite->create<ConstantOp>(
      loc, IntegerAttr::get(rewrite->getI64Type(), dim));
  return rewrite->create<DimOp>(loc, operand, dim_const);
}

llvm::SmallVector<Value> buildTensorDims(mlir::Value operand,
                                         RewriterBase* rewrite) {
  auto tensor = llvm::dyn_cast_or_null<ShapedType>(operand.getType());
  CHECK(llc::MLIR_PASS, tensor);
  auto rank = tensor.getRank();
  auto ranks = SmallVector<Value>();
  for (int i{}; i < rank; i++) {
    ranks.push_back(buildTensorDim(operand, rewrite, i));
  }
  return ranks;
}

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

}  // namespace mlir::llh