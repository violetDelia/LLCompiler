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

#include "llcompiler/Conversion/LLHToTensor/LLHToTensor.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/IR/LLHTypesImpl.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Builder.h"
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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
#define GEN_PASS_DEF_CONVERTLLHTOSHAPEPASS
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// operation lowing
//===----------------------------------------------------------------------===//
struct LLHDimOpToShape : public OpConversionPattern<DimOp> {
  using OpConversionPattern<DimOp>::OpConversionPattern;
  LogicalResult match(DimOp op) const final { return llvm::success(); }

  void rewrite(DimOp op, OpAdaptor adaptor,
               ConversionPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    auto input = op.getInput();
    auto dim = op.getDim();
    auto attrs = op->getAttrs();
    auto res = op->getResult(0);
    auto index_dim = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), dim);
    auto new_dim = rewriter.create<shape::DimOp>(
        loc, rewriter.getIndexType(), ::mlir::ValueRange{input, index_dim},
        attrs);
    auto index_out = rewriter.create<mlir::arith::IndexCastOp>(
        loc, op->getResultTypes(), new_dim);
    rewriter.replaceOp(op, index_out);
  }
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
LLC_DEFINR_CONVERSION_PASS(
    ConvertLLHToShape, {LLC_ADD_CONVERSION(LLHDimOpToShape)},
    {
      target.addLegalDialect<mlir::arith::ArithDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::index::IndexDialect>();
      target.addLegalDialect<mlir::shape::ShapeDialect>();
      target.addIllegalOp<DimOp>();
    },
    {
      auto type_replace = [](Type type) { return type; };
      auto int_replace = [](IntegerType type) { return type; };
      auto index_replace = [](IndexType type) { return type; };
      converter.addConversion(type_replace);
      converter.addConversion(int_replace);
      converter.addConversion(index_replace);
    })
