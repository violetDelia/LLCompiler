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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_UTILS_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_UTILS_H_

#include <cstddef>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/RewritePattern.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::llh {
llh::DimOp buildTensorDim(mlir::Value operand, LLCPatternRewriter* rewrite,
                          size_t dim);
llvm::SmallVector<Value> buildTensorDims(mlir::Value operand,
                                         LLCPatternRewriter* rewrite);
RankedTensorType cloneTensorWithEncoding(RankedTensorType type,
                                         EncodingAttr encoding);
}  // namespace mlir::llh

#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_UTILS_H_
