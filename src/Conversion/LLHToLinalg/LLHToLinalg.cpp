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
#include "llcompiler/Support/MlirUtility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#define GEN_PASS_DEF_CONVERTLLHTOLINALGPASS
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
//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct LLHScalarCastOPToLinalg : public OpConversionPattern<ScalarCastOP> {
  using OpConversionPattern<ScalarCastOP>::OpConversionPattern;

  LogicalResult match(ScalarCastOP op) const final { return success(); }

  void rewrite(ScalarCastOP op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    Loc_And_Context;;
    auto input = op.getInput();
    auto input_type = input.getType();
    if (isa<IntegerType, FloatType>(input_type)) {
      auto result_type = cast<ShapedType>(op.getType()).getElementType();
      auto empty = Tensor_Empty(llvm::SmallVector<int64_t>{1}, result_type);
      auto fill = rewriter.replaceOpWithNewOp<linalg::FillOp>(
          op, ValueRange{input}, ValueRange{empty});

    } else {
      UNIMPLEMENTED(llc::MLIR_PASS);
    }
  }
};

}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

LLC_DEFINE_CONVERSION_PASS(
    ConvertLLHToLinalg, { LLC_ADD_CONVERSION(LLHScalarCastOPToLinalg); },
    {
      target.addIllegalOp<ScalarCastOP>();
      target.addLegalDialect<mlir::linalg::LinalgDialect>();
      target.addLegalDialect<mlir::arith::ArithDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::tensor::TensorDialect>();
    },
    {
      auto shaped_repalce = [](ShapedType type) { return type; };
      auto int_repalce = [](IntegerType type) { return type; };
      auto ranked_tensor_replace = [](RankedTensorType type) { return type; };
      converter.addConversion(ranked_tensor_replace);
      converter.addConversion(shaped_repalce);
      converter.addConversion(int_repalce);
    })
