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

#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
namespace mlir::LLVM::ex {
#define GEN_PASS_DEF_ADAPTENTRYPARMSFORENGINEPASS
#include "llcompiler/Dialect/LLVMExtension/Transforms/Passes.h.inc"
}  // namespace mlir::LLVM::ex
using namespace ::mlir;
using namespace ::mlir::LLVM;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
namespace {

llvm::SmallVector<size_t> analysisOriginalMemrefStart(
    llvm::ArrayRef<Type>& params, size_t range_start) {
  llvm::SmallVector<size_t> original_memref_start;
  bool is_memref_begin = false;
  for (size_t i = range_start; i < params.size(); i++) {
    auto type = params[i];
    if (llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
      if (is_memref_begin) continue;
      is_memref_begin = true;
      original_memref_start.push_back(i);
    } else {
      is_memref_begin = false;
    }
  }
  return original_memref_start;
}

llvm::SmallVector<size_t> analysisOriginalMemrefRank(
    llvm::ArrayRef<Type>& params, size_t range_start) {
  llvm::SmallVector<size_t> original_memref_rank;
  bool pass_by_memref = false;
  size_t begin = 0;
  size_t end = 0;
  for (int i = range_start; i < params.size(); ++i) {
    auto type = params[i];
    if (llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
      if (pass_by_memref) {
        end = i;
        continue;
      }
      pass_by_memref = true;
      if (!(begin == end && begin == 0)) {
        original_memref_rank.push_back((end - begin - 2) / 2);
      }
      begin = i;
    } else {
      pass_by_memref = false;
      end = i;
    }
  }
  original_memref_rank.push_back((params.size() - begin - 3) / 2);
  return original_memref_rank;
}

void transformFunctionType(LLVM::LLVMFuncOp& enter_func, size_t memref_nums) {
  auto new_params = llvm::SmallVector<Type>();
  auto contxt = enter_func->getContext();
  new_params.push_back(LLVM::LLVMPointerType::get(contxt));
  for (auto _ : llvm::index_range(0, memref_nums)) {
    new_params.push_back(LLVM::LLVMPointerType::get(contxt));
    new_params.push_back(LLVM::LLVMPointerType::get(contxt));
    new_params.push_back(LLVM::LLVMPointerType::get(contxt));
    new_params.push_back(LLVM::LLVMPointerType::get(contxt));
    new_params.push_back(LLVM::LLVMPointerType::get(contxt));
  }
  auto new_enter_type = LLVM::LLVMFunctionType::get(
      enter_func.getFunctionType().getReturnType(), new_params);
  enter_func.setType(new_enter_type);
  enter_func.removeArgAttrsAttr();
}

void transformMemrefPtrs(size_t index, IRRewriter* rewriter, Block& block) {
  auto context = rewriter->getContext();
  auto base_prt = block.getArgument(index);
  auto data_ptr = block.getArgument(index + 1);
  auto loc = base_prt.getLoc();
  auto new_base_prt = block.addArgument(
      LLVM::LLVMPointerType::get(
          context,
          cast<LLVM::LLVMPointerType>(base_prt.getType()).getAddressSpace()),
      loc);
  rewriter->replaceAllUsesWith(base_prt, new_base_prt);
  auto new_data_prt = block.addArgument(
      LLVM::LLVMPointerType::get(
          context,
          cast<LLVM::LLVMPointerType>(base_prt.getType()).getAddressSpace()),
      loc);
  rewriter->replaceAllUsesWith(data_ptr, new_data_prt);
}

void transformMemrefOffset(size_t index, IRRewriter* rewriter, Block& block) {
  auto context = rewriter->getContext();
  auto offset = block.getArgument(index);
  auto loc = offset.getLoc();
  auto new_offset_ptr =
      block.addArgument(LLVM::LLVMPointerType::get(context), loc);
  auto const_op = rewriter->create<LLVM::ConstantOp>(
      loc, rewriter->getI64Type(), rewriter->getIndexAttr(0));
  auto get_element_ptr = rewriter->create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(
          context, cast<LLVM::LLVMPointerType>(new_offset_ptr.getType())
                       .getAddressSpace()),
      rewriter->getI64Type(), new_offset_ptr, ArrayRef<Value>{const_op}, true);
  auto new_offset =
      rewriter->create<LLVM::LoadOp>(loc, offset.getType(), get_element_ptr);
  block.push_front(new_offset);
  block.push_front(get_element_ptr);
  block.push_front(const_op);
  rewriter->replaceAllUsesWith(offset, new_offset);
}

void transformElementsRangeToPtr(size_t strat, size_t nums,
                                 IRRewriter* rewriter, Block& block) {
  auto context = rewriter->getContext();
  auto new_arg = block.addArgument(LLVM::LLVMPointerType::get(context),
                                   block.getArgument(0).getLoc());
  for (size_t i = 0; i < nums; ++i) {
    auto element = block.getArgument(strat + i);
    auto loc = element.getLoc();
    auto const_op = rewriter->create<LLVM::ConstantOp>(
        loc, rewriter->getI64Type(), rewriter->getIndexAttr(i));
    auto get_element_ptr = rewriter->create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(
            context,
            cast<LLVM::LLVMPointerType>(new_arg.getType()).getAddressSpace()),
        element.getType(), new_arg, ArrayRef<Value>{const_op}, true);
    auto new_element =
        rewriter->create<LLVM::LoadOp>(loc, element.getType(), get_element_ptr);
    block.push_front(new_element);
    block.push_front(get_element_ptr);
    block.push_front(const_op);
    rewriter->replaceAllUsesWith(element, new_element);
  }
}

void transformSymbolInt(size_t start, size_t nums, IRRewriter* rewriter,
                        Block& block) {
  auto context = rewriter->getContext();
  for (auto i : llvm::index_range(start, nums)) {
    auto new_arg = block.addArgument(LLVM::LLVMPointerType::get(context),
                                     block.getArgument(0).getLoc());
  }
}

void transformBlockArgs(Block& block, IRRewriter* rewriter,
                        llvm::SmallVector<size_t>& original_memref_start,
                        llvm::SmallVector<size_t>& original_memref_rank,
                        size_t symbol_int_nums) {
  auto context = block.back().getContext();
  auto loc = block.back().getLoc();
  auto arg_size = block.getNumArguments();
  auto memref_nums = original_memref_start.size();
  if (symbol_int_nums != 0) {
    transformElementsRangeToPtr(0, symbol_int_nums, rewriter, block);
  }
  for (int i = 0; i < memref_nums; ++i) {
    auto index = original_memref_start[i];
    auto rank = original_memref_rank[i];
    transformMemrefPtrs(index, rewriter, block);
    index += 2;
    transformMemrefOffset(index, rewriter, block);
    index += 1;
    transformElementsRangeToPtr(index, rank, rewriter, block);
    index += rank;
    transformElementsRangeToPtr(index, rank, rewriter, block);
  }
  block.eraseArguments(0, arg_size);
}

void transformBlockArgsFinal(Block& block, IRRewriter* rewriter) {
  auto context = block.back().getContext();
  auto loc = block.back().getLoc();
  auto arg_size = block.getNumArguments();
  auto new_arg_ptr =
      block.addArgument(LLVM::LLVMPointerType::get(context), loc);
  for (int i = 0; i < arg_size; ++i) {
    auto arg_ptr = block.getArgument(i);
    auto const_op = rewriter->create<LLVM::ConstantOp>(
        loc, rewriter->getI64Type(), rewriter->getIndexAttr(i));
    auto get_element_ptr = rewriter->create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(
            context, cast<LLVM::LLVMPointerType>(new_arg_ptr.getType())
                         .getAddressSpace()),
        LLVM::LLVMPointerType::get(
            context,
            cast<LLVM::LLVMPointerType>(arg_ptr.getType()).getAddressSpace()),
        new_arg_ptr, ArrayRef<Value>{const_op}, true);
    auto new_arg = rewriter->create<LLVM::LoadOp>(
        loc,
        LLVM::LLVMPointerType::get(
            context, cast<LLVM::LLVMPointerType>(new_arg_ptr.getType())
                         .getAddressSpace()),
        get_element_ptr);
    block.push_front(new_arg);
    block.push_front(get_element_ptr);
    block.push_front(const_op);
    rewriter->replaceAllUsesWith(arg_ptr, new_arg);
  }
  block.eraseArguments(0, arg_size);
}

void transformFunctionTypeFinal(LLVM::LLVMFuncOp& enter_func) {
  auto new_params = llvm::SmallVector<Type>();
  auto contxt = enter_func->getContext();
  new_params.push_back(LLVM::LLVMPointerType::get(contxt));
  auto new_enter_type = LLVM::LLVMFunctionType::get(
      enter_func.getFunctionType().getReturnType(), new_params);
  enter_func.setType(new_enter_type);
}
}  // namespace
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// pattern population
//===----------------------------------------------------------------------===//
void populateAdaptEntryParmsForEnginePassPatterns(RewritePatternSet& patterns) {
  auto context = patterns.getContext();
  // patterns.add<AdaptReturnOp>(context);
}

//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//

struct AdaptEntryParmsForEnginePass
    : ::LLVM::ex::impl::AdaptEntryParmsForEnginePassBase<
          AdaptEntryParmsForEnginePass> {
  void runOnOperation() override;
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass implement
//===----------------------------------------------------------------------===//

void AdaptEntryParmsForEnginePass::runOnOperation() {
  LLC_RUN_IN_PASS
  auto module = getOperation();
  IRRewriter rewriter(module->getContext());
  auto enter_func = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("main");
  CHECK(llc::MLIR_PASS, enter_func);
  auto enter_type = enter_func.getFunctionType();
  auto& block = enter_func.getFunctionBody().getBlocks().front();
  auto params = enter_type.getParams();
  auto symbol_int_nums = 0;
  if (enter_func->hasAttr(llc::SymbolIntArgNumsAttr)) {
    symbol_int_nums = llc::get_symbol_int_arg_nums_attr(enter_func);
  }
  auto original_memref_start =
      analysisOriginalMemrefStart(params, symbol_int_nums);
  auto original_memref_rank =
      analysisOriginalMemrefRank(params, symbol_int_nums);
  CHECK_EQ(llc::MLIR_PASS, original_memref_start.size(),
           original_memref_rank.size());
  transformBlockArgs(block, &rewriter, original_memref_start,
                     original_memref_rank, symbol_int_nums);
  transformFunctionType(enter_func, original_memref_rank.size());
  transformBlockArgsFinal(block, &rewriter);
  transformFunctionTypeFinal(enter_func);
  LLC_RUN_OUT_PASS
}
