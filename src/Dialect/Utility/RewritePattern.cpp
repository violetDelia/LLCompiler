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
#include "llcompiler/Dialect/Utility/RewritePattern.h"

#include "llcompiler/Dialect/LLH/Utils/Broadcast.h"
#include "llcompiler/Dialect/LLH/Utils/InferSymbol.h"
#include "llcompiler/Support/Logger.h"
namespace mlir {
// 不要再这个方法里面创建非symbolOp
void LLHPatternRewriter::processWileBuildOperation(Operation *op) {
  llh::checkAndInferSymbol(op);
  // llh::checkBroadcast(op);
}

bool LLHPatternRewriter::canRecoverFromRewriteFailure() const { return false; }

}  // namespace mlir
