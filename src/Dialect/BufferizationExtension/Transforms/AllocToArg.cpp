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

#include "llcompiler/Dialect/BufferizationExtension/Transforms/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llcompiler/Support/MlirUtility.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir::bufferization::ex {
#define GEN_PASS_DEF_ALLOCTOARGPASS
#include "llcompiler/Dialect/BufferizationExtension/Transforms/Passes.h.inc"
}  // namespace mlir::bufferization::ex
using namespace ::mlir;
using namespace ::mlir::bufferization;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//
void allocToArg(func::FuncOp funcOp) {
  auto &blocks = funcOp.getFunctionBody().getBlocks();
  if (blocks.size() != 1) {
    return;
  }
  Block &bodyBlock = blocks.front();
  auto returnOp = llvm::cast<func::ReturnOp>(bodyBlock.getTerminator());

  IRRewriter rewriter(funcOp.getContext());
  BitVector resultsToErase(funcOp.getNumResults());
  Location loc = returnOp.getLoc();

  for (auto [i, result] : llvm::enumerate(returnOp.getOperands())) {
    Operation *resultDef = result.getDefiningOp();
    Type resultTy = result.getType();

    // Case: plain alloc.
    if (auto allocOp = llvm::dyn_cast_or_null<memref::AllocOp>(resultDef)) {
      resultsToErase.set(i);
      auto attrs = funcOp.getResultAttrDict(i);
      funcOp.insertArgument(funcOp.getNumArguments(), resultTy, attrs, loc);
      rewriter.replaceOp(allocOp, funcOp.getArguments().back());
      continue;
    }

    // Case: shape-expanded alloc.
    if (auto expandOp =
            llvm::dyn_cast_or_null<memref::ExpandShapeOp>(resultDef)) {
      Operation *expandDef = expandOp.getOperand(0).getDefiningOp();
      if (auto allocOp = llvm::dyn_cast_or_null<memref::AllocOp>(expandDef)) {
        resultsToErase.set(i);
        auto attrs = funcOp.getResultAttrDict(i);
        funcOp.insertArgument(funcOp.getNumArguments(), resultTy, attrs, loc);

        // Collapse buffer argument to replace possible uses of the unexpanded
        // buffer.
        rewriter.setInsertionPoint(allocOp);
        Value arg = funcOp.getArguments().back();
        Value collapsedArg =
            Mem_CollapseShape(arg, expandOp.getReassociationIndices());

        // Replace alloc and its expansion.
        rewriter.replaceOp(allocOp, collapsedArg);
        rewriter.replaceOp(expandOp, arg);
        continue;
      }
    }
    return;
  }

  funcOp.eraseResults(resultsToErase);
  returnOp->eraseOperands(resultsToErase);
}
//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct MemrefAllocToArg : public OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern<mlir::memref::CopyOp>::OpRewritePattern;

  LogicalResult match(mlir::memref::CopyOp op) const {
    auto target = op.getTarget();
    if (!isa<BlockArgument>(target)) return llvm::failure();
    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (!func) return llvm::failure();
    if (!func->hasAttr(llc::EntranceAttr)) return llvm::failure();
    auto source = op.getSource();
    if (getRootAllocOp(source) == nullptr) return llvm::failure();
    return llvm::success();
  }

  void rewrite(mlir::memref::CopyOp op, PatternRewriter &rewriter) const {
    auto source = op.getSource();
    auto target = op.getTarget();
    auto alloc = llvm::cast<memref::AllocOp>(getRootAllocOp(source));
    rewriter.replaceAllUsesWith(alloc, target);
    rewriter.eraseOp(op);
  }

  Operation *getRootAllocOp(Value source) const {
    if (isa<BlockArgument>(source)) return nullptr;
    auto maybe_root = source.getDefiningOp();
    if (dyn_cast<memref::AllocOp>(maybe_root)) return maybe_root;
    return nullptr;
  }
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
using namespace ::mlir::bufferization::ex::impl;
LLC_DEFINE_PASS(AllocToArg, { LLC_ADD_PATTERN(MemrefAllocToArg); }, {}, {})
