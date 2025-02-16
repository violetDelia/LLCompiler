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
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Conversion/LLHToArith/LLHToArith.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#define GEN_PASS_DEF_CONVERTLLHTOMATHPASS
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
bool check_unary_legal(Operation* op) {
  auto res_type = op->getResult(0).getType();
  auto input_type = op->getOperand(0).getType();
  if (isa<IntegerType>(input_type) && isa<IntegerType>(res_type)) return false;
  return true;
}
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct LLHSqrtOpToMath : public OpConversionPattern<SqrtOp> {
  using OpConversionPattern<SqrtOp>::OpConversionPattern;

  LogicalResult match(SqrtOp op) const final { return success(); }

  void rewrite(SqrtOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    op->dump();
    auto loc = op->getLoc();
    auto input = op.getInput();
    auto input_type = input.getType();
    auto fast_math = arith::FastMathFlagsAttr::get(
        op->getContext(), ::mlir::arith::FastMathFlags::none);
    if (isa<IntegerType>(input_type)) {
      CHECK(llc::MLIR_PASS, !llvm::cast<IntegerType>(input_type).isUnsigned())
          << "Invalid input";
      auto f32 = rewriter.getF32Type();
      auto si_to_fp = rewriter.create<arith::SIToFPOp>(loc, f32, input);

      auto sqrt = rewriter.create<math::SqrtOp>(loc, f32, si_to_fp, fast_math);
      rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, input_type, sqrt);
    } else {
      rewriter.replaceOpWithNewOp<math::SqrtOp>(op, input_type, input,
                                                fast_math);
    }
  }
};
//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//

}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

LLC_DEFINE_CONVERSION_PASS(
    ConvertLLHToMath, { LLC_ADD_CONVERSION(LLHSqrtOpToMath); },
    {
      target.addDynamicallyLegalOp<SqrtOp>(check_unary_legal);
      target.addLegalDialect<mlir::math::MathDialect>();
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
