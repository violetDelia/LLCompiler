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

#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/InferSymbol.h"

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
namespace mlir::llh {

namespace {

llvm::StringRef buildBinaryOpSymbol(Operation* op) {
  auto analysis = SymbolAnalysis::getInstance(op);
  auto lhs = op->getOperand(0);
  if (!SymbolAnalysis::hasSymbolAttr(lhs)) return SymbolAnalysis::UNKOW_SYMBOL;
  auto rhs = op->getOperand(1);
  if (!SymbolAnalysis::hasSymbolAttr(rhs)) return SymbolAnalysis::UNKOW_SYMBOL;
  llvm::SmallString<4> symbol;
  llh::SymbolRelation relation;
  if (isa<MulOp, arith::MulIOp>(op)) {
    relation = SymbolRelation::Mul;
  } else if (isa<AddOp, arith::AddIOp>(op)) {
    relation = SymbolRelation::Add;
  } else if (isa<SubOp, arith::SubIOp>(op)) {
    relation = SymbolRelation::Sub;
  } else if (isa<DivOp>(op)) {
    relation = SymbolRelation::FloorDiv;
  } else {
    UNIMPLEMENTED(llc::SymbolInfer);
  }
  auto new_symbol = analysis->buildNewSymbolWithRelation(
      analysis->getOrBuildSymbolAttrFrom(lhs),
      analysis->getOrBuildSymbolAttrFrom(rhs), relation);
  llc::add_symbol_attr(op, new_symbol.getSymName());
  return new_symbol.getSymName();
}

llvm::StringRef buildDimOpSymbol(Operation* op, Value shape_value,
                                 Value const_value) {
  LLHPatternRewriter builder(op->getContext());
  llvm::SmallVector<llvm::StringRef> shape_symbols =
      SymbolAnalysis::getEncodingShapes(shape_value);
  if (shape_symbols.empty()) return SymbolAnalysis::UNKOW_SYMBOL;
  auto dim = llh::getConstIntegerValue(const_value);
  llc::add_symbol_attr(op, shape_symbols[dim]);
  return shape_symbols[dim];
}

llvm::StringRef buildConstSymbol(Operation* op) {
  auto res = op->getResult(0);
  if (!llvm::isa<IndexType, IntegerType>(res.getType()))
    return SymbolAnalysis::UNKOW_SYMBOL;
  size_t val;
  if (isa<ConstantOp, arith::ConstantOp>(op)) {
    if (isa<IntegerType, IndexType>(op->getResult(0).getType())) {
      val = llvm::cast<IntegerAttr>(op->getAttr("value")).getInt();
    } else {
      UNIMPLEMENTED(llc::SymbolInfer);
    }
  } else {
    UNIMPLEMENTED(llc::UTILITY) << op->getName().getStringRef().str();
  }
  auto symbol_analysis = SymbolAnalysis::getInstance(op);
  auto symbol_dim_op = symbol_analysis->getOrBuildConstSymbol(val);
  llc::add_symbol_attr(op, symbol_dim_op.getSymName());
  return symbol_dim_op.getSymName();
}

}  // namespace

void checkAndInferSymbol(Operation* op) {
  if (!SymbolAnalysis::symbol_enable) return;
  auto symbol_op = llvm::dyn_cast_or_null<SymbolicInferShapeOpInterface>(op);
  if (symbol_op) {
    symbol_op.inferSymbolicShape();
    return;
  }
  if (SymbolAnalysis::isExtraSymbolAttrInferOp(op)) {
    SymbolAnalysis::getInstance(op)->BuildSymbolAttrFrom(op);
  }
}

bool SymbolAnalysis ::isExtraSymbolAttrInferOp(Operation* op) {
  return isa<DimOp, tensor::DimOp, arith::ConstantOp, arith::SubIOp,
             arith::AddIOp, arith::MulIOp>(op);
}

bool SymbolAnalysis ::isExtraSymbolEncodingInferOp(Operation* op) {
  return isa<memref::AllocaOp>(op);
}

bool SymbolAnalysis ::isExtraSymbolicInferOp(Operation* op) {
  return isExtraSymbolAttrInferOp(op) || isExtraSymbolEncodingInferOp(op);
}

bool SymbolAnalysis ::isSymbolicInferOp(Operation* op) {
  return isExtraSymbolicInferOp(op) || isa<SymbolicInferShapeOpInterface>(op);
}

llvm::StringRef SymbolAnalysis::BuildSymbolAttrFrom(Operation* op) {
  if (isa<DimOp, mlir::tensor::DimOp>(op)) {
    return buildDimOpSymbol(op, op->getOperand(0), op->getOperand(1));
  }
  if (isa<mlir::arith::ConstantIntOp, mlir::arith::ConstantOp,
          mlir::arith::ConstantIndexOp, llh::ConstantOp>(op)) {
    return buildConstSymbol(op);
  }
  if (isa<DivOp, MulOp, SubOp, AddOp, arith::SubIOp, arith::AddIOp,
          arith::MulIOp>(op)) {
    return buildBinaryOpSymbol(op);
  }
  return SymbolAnalysis::UNKOW_SYMBOL;
}

llvm::StringRef SymbolAnalysis::BuildSymbolAttrFrom(Value value) {
  auto op = value.getDefiningOp();
  return BuildSymbolAttrFrom(op);
}

}  // namespace mlir::llh
