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
#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Interfaces/SymbolShapeOpInterfaces.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::llh {

bool isLayoutSensitive(Operation* op) {
  return llvm::isa<llh::ConvOp, ConvBiasOp, MaxPoolOp>(op);
}

Value getElementNums(mlir::Value operand, LLHPatternRewriter* rewrite) {
  auto dims = buildTensorDims(operand, rewrite);
  return getElementNums(dims, rewrite);
}

Value getElementNums(llvm::SmallVector<Value> dims,
                     LLHPatternRewriter* rewrite) {
  CHECK(llc::UTILITY, dims.size() != 0);
  if (dims.size() == 1) return dims[0];
  Value element_nums = dims[0];
  auto loc = element_nums.getLoc();
  for (auto index : llvm::index_range(1, dims.size())) {
    element_nums =
        rewrite->create<MulOp>(loc, TypeRange{element_nums.getType()},
                               ValueRange{element_nums, dims[index]});
  }
  return element_nums;
}
bool isConstIntegerValue(Value value) {
  auto type = value.getType();
  if (!llvm::isa<IntegerType, IndexType>(type)) return false;
  if (!llvm::isa<BlockArgument>(value) &&
      isa<MulOp, SubOp, AddOp, DivOp>(value.getDefiningOp())) {
    return isConstIntegerValue(value.getDefiningOp()->getOperand(0)) &&
           isConstIntegerValue(value.getDefiningOp()->getOperand(1));
  }
  if (!llvm::isa<BlockArgument>(value) &&
      llvm::isa<DimOp>(value.getDefiningOp())) {
    auto dim_op = cast<DimOp>(value.getDefiningOp());
    auto dim_type = llvm::cast<RankedTensorType>(dim_op.getInput().getType());
    CHECK(llc::MLIR, isConstIntegerValue(dim_op.getDim()));
    return !dim_type.isDynamicDim(getConstIntegerValue(dim_op.getDim()));
  }
  if (llvm::isa<BlockArgument>(value)) return false;
  auto op = value.getDefiningOp();
  if (isa<TorchSymbolicIntOp>(op)) return false;
  if (llvm::isa<mlir::arith::ConstantOp>(op)) return true;
  if (llvm::isa<ConstantOp>(op)) {
    auto constant_op = llvm::cast<ConstantOp>(op);
    return llvm::isa<IntegerAttr>(constant_op.getValueAttr());
  }
  UNIMPLEMENTED(llc::UTILITY) << "unsupport check operator is const: "
                              << op->getName().getStringRef().str();
  return false;
}

int64_t getConstIntegerValue(Value value) {
  CHECK(llc::UTILITY, !isa<BlockArgument>(value));
  auto type = value.getType();
  if (!llvm::isa<IntegerType, IndexType>(type)) FATAL(llc::MLIR);
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
  if (llvm::isa<arith::ConstantOp>(op)) {
    auto constant_op = llvm::cast<arith::ConstantOp>(op);
    if (!llvm::isa<IntegerAttr>(constant_op.getValueAttr())) FATAL(llc::MLIR);
    return llvm::cast<IntegerAttr>(constant_op.getValueAttr()).getInt();
  }
  if (isa<MulOp>(op)) {
    return getConstIntegerValue(op->getOperand(0)) *
           getConstIntegerValue(op->getOperand(1));
  }
  if (isa<SubOp>(op)) {
    return getConstIntegerValue(op->getOperand(0)) -
           getConstIntegerValue(op->getOperand(1));
  }
  if (isa<AddOp>(op)) {
    return getConstIntegerValue(op->getOperand(0)) +
           getConstIntegerValue(op->getOperand(1));
  }
  if (isa<DivOp>(op)) {
    return getConstIntegerValue(op->getOperand(0)) /
           getConstIntegerValue(op->getOperand(1));
  }
  UNIMPLEMENTED(llc::UTILITY) << "unsupport get operator const value: "
                              << op->getName().getStringRef().str();
}

Layout getLayoutFromGloabalLayout(Layout global_layout, int64_t rank) {
  if (rank == 3) {
    if (global_layout == Layout::NCHW) return Layout::NCW;
    if (global_layout == Layout::NHWC) return Layout::NWC;
  } else if (rank == 4) {
    if (global_layout == Layout::NCHW) return Layout::NCHW;
    if (global_layout == Layout::NHWC) return Layout::NHWC;
  } else {
    UNIMPLEMENTED(llc::UTILITY);
  }
}
Layout getWeightLayoutFromGloabalLayout(Layout global_layout, int64_t rank) {
  if (rank == 3) {
    if (global_layout == Layout::NCHW) return Layout::FCW;
    if (global_layout == Layout::NHWC) return Layout::WCF;
  } else if (rank == 4) {
    if (global_layout == Layout::NCHW) return Layout::FCHW;
    if (global_layout == Layout::NHWC) return Layout::FHWC;
  } else {
    UNIMPLEMENTED(llc::UTILITY);
  }
}

}  // namespace mlir::llh
