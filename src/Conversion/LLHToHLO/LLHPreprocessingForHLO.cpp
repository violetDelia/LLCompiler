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
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "llcompiler/Conversion/LLHToHLO/LLHToHLO.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
namespace mlir {
#define GEN_PASS_DEF_LLHPREPROCESSINGFORHLOPASS
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
struct LLHReluOpSwitch : public LLHOpRewritePattern<ReluOp> {
  using LLHOpRewritePattern<ReluOp>::LLHOpRewritePattern;
  LogicalResult match(ReluOp op) const final { return llvm::success(); }
  void rewrite(ReluOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res.getType());
    auto res_ele_type = res_type.getElementType();
    DenseElementsAttr value;

    if (isa<IntegerType>(res_ele_type)) {
      value = SplatElementsAttr::get(RankedTensorType::get({1}, res_ele_type),
                                     IntegerAttr::get(res_ele_type, 0));
    } else if (isa<FloatType>(res_ele_type)) {
      value = SplatElementsAttr::get(RankedTensorType::get({1}, res_ele_type),
                                     FloatAttr::get(res_ele_type, 0));
    } else {
      UNIMPLEMENTED(llc::MLIR_PASS);
    }
    auto zore = rewriter.create<ConstantOp>(loc, value);
    rewriter.replaceOpWithNewOp<MaxOp>(op, TypeRange{res_type},
                                       ValueRange{input, zore},
                                       op->getAttrDictionary().getValue());
  }
};

struct LLHExtractOpSwitch : public LLHOpRewritePattern<ExtractOp> {
  using LLHOpRewritePattern<ExtractOp>::LLHOpRewritePattern;
  LogicalResult match(ExtractOp op) const final {
    auto input = op.getInput();
    auto input_type = llc::getRankTensorFrom(input);
    auto rank = input_type.getRank();
    // if (rank == 1) return llvm::failure();
    return llvm::success();
  }

  void rewrite(ExtractOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    auto one = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    auto zore = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    auto input = op.getInput();
    auto input_type = llc::getRankTensorFrom(input);
    auto rank = input_type.getRank();
    auto slice_out_shape = llc::getShapeFrom(input_type);
    auto dims = llh::buildTensorDims(input, &rewriter);
    auto index = op.getIndex();
    llvm::SmallVector<Value> start(rank, zore);
    start[0] = index;
    auto end_index = rewriter.create<AddOp>(loc, TypeRange{index.getType()},
                                            ValueRange{index, one});
    dims[0] = end_index;
    llvm::SmallVector<Value> stride(rank, one);
    slice_out_shape[0] = 1;
    auto slice_out_type = input_type.clone(slice_out_shape);
    auto slice = rewriter.create<StrideSliceOp>(loc, slice_out_type, input,
                                                start, dims, stride);
    if (rank != 1) {
      dims.erase(dims.begin());
      auto reshape = rewriter.create<ReshapeOp>(loc, op.getType(), slice, dims);
      rewriter.replaceOp(op, reshape);
    } else {
      rewriter.replaceOp(op, slice);
    }
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
LLC_DEFINE_PASS(LLHPreprocessingForHLO,
                {
                  LLC_ADD_PATTERN(LLHReluOpSwitch);
                  LLC_ADD_PATTERN(LLHExtractOpSwitch);
                },
                {}, {})
