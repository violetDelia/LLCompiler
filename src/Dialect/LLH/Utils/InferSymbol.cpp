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

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
namespace mlir::llh {

void checkAndInferSymbol(Operation* op) {
  if (!SymbolAnalysis::symbol_enable) return;
  auto symbol_op = llvm::dyn_cast_or_null<SymbolicInferShapeOpInterface>(op);
  if (symbol_op) {
    symbol_op.inferSymbolicShape();
    return;
  }
  if (SymbolAnalysis::isExtraSymbolicInferOp(op)) {
    SymbolAnalysis::getInstance(op)->getOrBuildSymbolAttr(op);
  }
}
}  // namespace mlir::llh