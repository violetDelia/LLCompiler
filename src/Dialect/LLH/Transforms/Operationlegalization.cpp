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

#include <cstddef>
#include <cstdint>
#include <regex>
#include <string>

#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Transforms/Passes.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::llh {
#define GEN_PASS_DEF_OPERATIONLEGALIZATIONPASS
#include "llcompiler/Dialect/LLH/Transforms/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

// op type only int/float
ConstantOp buildConstTensorFromScalar(ConstantOp op,
                                      LLHPatternRewriter* rewriter,
                                      Operation* user) {
  auto user_type =
      llvm::dyn_cast_or_null<ShapedType>(user->getResult(0).getType());
  CHECK(llc::MLIR_PASS, user_type);
  auto user_ele_type = user_type.getElementType();
  auto tensor_type = RankedTensorType::get({1}, user_ele_type);
  auto const_type = op->getResult(0).getType();
  auto loc = op->getLoc();
  DenseElementsAttr new_value;
  if (user_ele_type.isInteger()) {
    if (const_type.isInteger()) {
      auto data_attr = llvm::dyn_cast_or_null<IntegerAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      new_value = DenseElementsAttr::get(tensor_type, {(*data.getRawData())});
    } else {
      auto data_attr = llvm::dyn_cast_or_null<FloatAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      auto int_attr = IntegerAttr::get(user_ele_type, data.convertToFloat());
      new_value = DenseElementsAttr::get(tensor_type, {int_attr.getValue()});
    }
  } else {
    if (const_type.isInteger()) {
      auto data_attr = llvm::dyn_cast_or_null<IntegerAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      auto float_attr = FloatAttr::get(user_ele_type, *data.getRawData());
      new_value = DenseElementsAttr::get(tensor_type, {float_attr.getValue()});
    } else {
      auto data_attr = llvm::dyn_cast_or_null<FloatAttr>(op.getValueAttr());
      CHECK(llc::MLIR_PASS, data_attr);
      auto data = data_attr.getValue();
      auto float_attr = FloatAttr::get(user_ele_type, data.convertToFloat());
      new_value = DenseElementsAttr::get(tensor_type, {float_attr.getValue()});
    }
  }
  return rewriter->create<ConstantOp>(loc, new_value);
}

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//

template <class Op>
struct ResultScaleRefine : public LLHOpRewritePattern<Op> {
  using LLHOpRewritePattern<Op>::LLHOpRewritePattern;
  LogicalResult match(Op op) const final {
    for (auto res : op->getResults()) {
      auto tensor = llc::getRankTensorFrom(res);
      if (tensor.getRank() == 0) return llvm::success();
    }
    return llvm::failure();
  }
  void rewrite(Op op, LLHPatternRewriter& rewriter) const final {
    for (auto res : op->getResults()) {
      auto tensor = llc::getRankTensorFrom(res);
      auto new_tensor = RankedTensorType::get({1}, tensor.getElementType());
      res.setType(new_tensor);
      for (auto user : res.getUsers()) {
        if (isa<func::ReturnOp>(user)) {
          auto return_op = cast<func::ReturnOp>(user);
          auto operands = return_op->getOperands();
          auto index = 0;
          for (auto operand : operands) {
            if (operand.getDefiningOp() == op.getOperation()) {
              break;
            }
            index++;
          }
          auto func = return_op->template getParentOfType<func::FuncOp>();
          auto func_type = func.getFunctionType();
          auto input = func_type.getInputs();
          auto output = func_type.getResults();
          llvm::SmallVector<Type> new_output;
          for (auto output_type : output) {
            new_output.push_back(output_type);
          }
          new_output[index] = new_tensor;
          auto new_func_type = rewriter.getFunctionType(input, new_output);
          func.setFunctionType(new_func_type);
        }
      }
    }
  }
};

struct LLHBroadcastOpRefine : public LLHOpRewritePattern<BroadCastToOp> {
  using LLHOpRewritePattern<BroadCastToOp>::LLHOpRewritePattern;
  LogicalResult match(BroadCastToOp op) const final {
    auto cast_dims = op.getCastDims();
    auto res = op.getResult();
    auto res_type = llc::getRankTensorFrom(res);
    if (res_type.getRank() == cast_dims.size()) return llvm::failure();
    return llvm::success();
  }

  void rewrite(BroadCastToOp op, LLHPatternRewriter& rewriter) const final {
    auto loc = op->getLoc();
    auto input = op.getInput();
    auto cast_dims = op.getCastDims();
    auto res = op.getResult();
    auto dims = op.getOutShapes();
    auto res_tensor = llc::getRankTensorFrom(res);
    auto ele_type = res_tensor.getElementType();
    auto res_rank = res_tensor.getRank();
    auto new_cast_dims = llvm::SmallVector<int64_t>();
    auto one = rewriter.create<ConstantOp>(
        loc, IntegerAttr::get(rewriter.getI64Type(), 1));
    auto reshape_dims = llvm::SmallVector<Value>(res_rank, one);
    auto reshape_res_shapes = llvm::SmallVector<int64_t>(res_rank, 1);
    for (int i = 0; i < res_rank; i++) {
      new_cast_dims.push_back(i);
    }
    for (auto [i, cast_dim] : llvm::enumerate(cast_dims)) {
      reshape_dims[cast_dim] = llh::buildTensorDim(input, &rewriter, i);
      if (llh::isConstIntegerValue(reshape_dims[cast_dim])) {
        reshape_res_shapes[cast_dim] =
            llh::getConstIntegerValue(reshape_dims[cast_dim]);
      } else {
        reshape_res_shapes[cast_dim] = ShapedType::kDynamic;
      }
    }
    auto reshape_res_type = RankedTensorType::get(reshape_res_shapes, ele_type);
    auto reshape =
        rewriter.create<ReshapeOp>(loc, reshape_res_type, input, reshape_dims);
    auto new_braodcast = rewriter.create<BroadCastToOp>(
        loc, res_tensor, reshape, dims, new_cast_dims, DenseI64ArrayAttr(),
        DenseI64ArrayAttr());
    rewriter.replaceOp(op, new_braodcast);
  }
};

struct LLHAddSymbolIntArgNumsAttr : public LLHOpRewritePattern<func::FuncOp> {
  using LLHOpRewritePattern<func::FuncOp>::LLHOpRewritePattern;
  LogicalResult match(func::FuncOp op) const final {
    if (!op->hasAttr(llc::EntranceAttr)) return llvm::failure();
    if (op->hasAttr(llc::SymbolIntArgNumsAttr)) return llvm::failure();
    return llvm::success();
  }

  void rewrite(func::FuncOp op, LLHPatternRewriter& rewriter) const final {
    auto func_type = op.getFunctionType();
    auto input_types = func_type.getInputs();
    int symbol_int_nums = 0;
    for (auto type : input_types) {
      if (llvm::isa<IntegerType>(type))
        symbol_int_nums++;
      else
        break;
    }
    llc::add_symbol_int_arg_nums_attr(op, symbol_int_nums);
  }
};

}  // namespace
using LLHWeightOpScaleRefine = ResultScaleRefine<WeightOp>;
using LLHExtractOpScaleRefine = ResultScaleRefine<ExtractOp>;
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
using namespace mlir::llh::impl;
LLC_DEFINR_PASS(Operationlegalization,
                {
                  LLC_ADD_PATTERN(LLHWeightOpScaleRefine);
                  LLC_ADD_PATTERN(LLHExtractOpScaleRefine);
                  LLC_ADD_PATTERN(LLHBroadcastOpRefine);
                  LLC_ADD_PATTERN(LLHAddSymbolIntArgNumsAttr);
                },
                {}, {})
