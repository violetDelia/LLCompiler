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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_COMMONREWRITE_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_COMMONREWRITE_H_
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/Operation.h"
namespace mlir::llh {
template <class BinaryOp>
struct SimplyBinaryOpInsertBraodcast : public LLHOpRewritePattern<BinaryOp> {
  using LLHOpRewritePattern<BinaryOp>::LLHOpRewritePattern;
  LogicalResult match(BinaryOp op) const final {
    return checkBinaryNeedBroadcast(op);
  }
  void rewrite(BinaryOp op, LLHPatternRewriter& rewriter) const final {
    insertBroadcastBeforeBinary<BinaryOp>(op, rewriter);
  }
};

template <class BinaryOp>
struct SimplyBinaryOpReshape : public LLHOpRewritePattern<BinaryOp> {
  using LLHOpRewritePattern<BinaryOp>::LLHOpRewritePattern;
  LogicalResult match(BinaryOp op) const final {
    return checkBinaryNeedReshape(op);
  }
  void rewrite(BinaryOp op, LLHPatternRewriter& rewriter) const final {
    insertReshapeBeforeBinary<BinaryOp>(op, rewriter);
  }
};

}  // namespace mlir::llh
#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_COMMONREWRITE_H_
