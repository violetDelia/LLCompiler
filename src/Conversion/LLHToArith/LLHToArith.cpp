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
#include "llcompiler/Conversion/LLHToArith/LLHToArith.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llcompiler/Support/MlirUtility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLHTOARITHPASS
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
  auto type = op->getResult(0).getType();
  return isa<RankedTensorType>(type);
}

bool check_binary_legal(Operation* op) {
  auto res_type = op->getResult(0).getType();
  auto lhs_type = op->getOperand(0).getType();
  auto rhs_type = op->getOperand(1).getType();
  if (isa<IntegerType>(lhs_type) && isa<IntegerType>(rhs_type) &&
      isa<IntegerType>(res_type))
    return false;
  return true;
}
bool check_unary_legal(Operation* op) {
  auto res_type = op->getResult(0).getType();
  auto input_type = op->getOperand(0).getType();
  if (isa<IntegerType>(input_type) && isa<IntegerType>(res_type)) return false;
  return true;
}
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
template <class FromOp, class ToOp>
struct SimplyBinaryOpLowing : public OpConversionPattern<FromOp> {
  using OpConversionPattern<FromOp>::OpConversionPattern;
  using OpAdaptor = typename FromOp::Adaptor;

  LogicalResult match(FromOp op) const final { return success(); }

  void rewrite(FromOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;
    auto res = op->getResult(0);
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto index_lhs = IndexCast(Index_Ty, lhs);
    auto index_rhs = IndexCast(Index_Ty, rhs);
    auto new_add = rewriter.create<ToOp>(loc, TypeRange{Index_Ty},
                                         ValueRange{index_lhs, index_rhs},
                                         op->getAttrDictionary().getValue());
    auto index_res = IndexCast(res.getType(), new_add);
    rewriter.replaceOp(op, index_res);
  }
};

}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
using LLHConstantOpToArith =
    SimplyFullLowing<llh::ConstantOp, arith::ConstantOp>;
using LLHMulOpToArith = SimplyBinaryOpLowing<llh::MulOp, arith::MulIOp>;
using LLHSubOpToArith = SimplyBinaryOpLowing<llh::SubOp, arith::SubIOp>;
using LLHAddOpToArith = SimplyBinaryOpLowing<llh::AddOp, arith::AddIOp>;
using LLHDivOpToArith = SimplyBinaryOpLowing<llh::DivOp, arith::DivUIOp>;
LLC_DEFINE_CONVERSION_PASS(
    ConvertLLHToArith,
    {
      LLC_ADD_CONVERSION(LLHConstantOpToArith);
      LLC_ADD_CONVERSION(LLHMulOpToArith);
      LLC_ADD_CONVERSION(LLHSubOpToArith);
      LLC_ADD_CONVERSION(LLHDivOpToArith);
      LLC_ADD_CONVERSION(LLHAddOpToArith);
    },
    {
      target.addDynamicallyLegalOp<ConstantOp>(check_const_legal);
      target.addDynamicallyLegalOp<AddOp>(check_binary_legal);
      target.addDynamicallyLegalOp<MulOp>(check_binary_legal);
      target.addDynamicallyLegalOp<DivOp>(check_binary_legal);
      target.addDynamicallyLegalOp<SubOp>(check_binary_legal);
      target.addLegalDialect<mlir::arith::ArithDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::index::IndexDialect>();
    },
    {
      auto shaped_repalce = [](ShapedType type) { return type; };
      auto int_repalce = [](IntegerType type) { return type; };
      auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
      converter.addConversion(ranked_tensor_replace);
      converter.addConversion(shaped_repalce);
      converter.addConversion(int_repalce);
    })
