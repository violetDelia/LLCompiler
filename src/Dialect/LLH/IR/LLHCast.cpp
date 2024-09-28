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
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::llh {

bool SymbolicCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  RankedTensorType input = llvm::dyn_cast<RankedTensorType>(inputs.front());
  RankedTensorType output = llvm::dyn_cast<RankedTensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // auto output_encoding =
  //     llvm::dyn_cast_or_null<EncodingAttr>(output.getEncoding());
  // if (!output_encoding) return false;
  return true;
}

}  // namespace mlir::llh
