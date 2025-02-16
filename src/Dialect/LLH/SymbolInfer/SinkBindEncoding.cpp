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

#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Macro.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::llh {
#define GEN_PASS_DEF_SINKBINDENCODINGPASS
#include "llcompiler/Dialect/LLH/SymbolInfer/Passes.h.inc"
}  // namespace mlir::llh
using namespace ::mlir;
using namespace ::mlir::llh;
namespace {
//===----------------------------------------------------------------------===//
// common func
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// transform patterns
//===----------------------------------------------------------------------===//
struct LLHEncodingBindOpSink : public LLHOpRewritePattern<EncodingBindOp> {
  using LLHOpRewritePattern::LLHOpRewritePattern;
  LogicalResult match(EncodingBindOp op) const final {
    auto operand = op.getOperand();
    if (isa<BlockArgument>(operand)) return llvm::failure();
    auto parent = operand.getDefiningOp();
    if (isa<mlir::bufferization::ToMemrefOp, mlir::bufferization::ToTensorOp>(
            parent))
      return llvm::success();
    return llvm::failure();
  }
  void rewrite(EncodingBindOp op, LLHPatternRewriter& rewriter) const final {
    auto parent = op.getOperand().getDefiningOp();
    auto val = parent->getResult(0);
    op.setOperand(val);
  }
};
}  // namespace
//===----------------------------------------------------------------------===//
// pass defination
//===----------------------------------------------------------------------===//
using namespace mlir::llh::impl;
LLC_DEFINE_PASS(SinkBindEncoding, { LLC_ADD_PATTERN(LLHEncodingBindOpSink); },
                {}, {})
