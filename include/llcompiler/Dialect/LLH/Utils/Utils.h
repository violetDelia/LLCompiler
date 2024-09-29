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
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir::llh {
bool isLayoutSensitive(Operation* op);

void checkAndInferSymbol(Operation* op);
llh::DimOp buildTensorDim(mlir::Value operand, RewriterBase* rewrite,
                          size_t dim);
llvm::SmallVector<Value> buildTensorDims(mlir::Value operand,
                                         RewriterBase* rewrite);
RankedTensorType cloneTensorWithEncoding(RankedTensorType type,
                                         EncodingAttr encoding);
size_t getConstIntegerValue(Value value);
bool isConstIntegerValue(Value value);
}  // namespace mlir::llh

#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_UTILS_UTILS_H_
