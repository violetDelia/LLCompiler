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

#ifndef INCLUDE_LLCOMPILER_CONVERSION_PASSES_H_
#define INCLUDE_LLCOMPILER_CONVERSION_PASSES_H_
#include "llcompiler/Conversion/LLHToArith/LLHToArith.h"
#include "llcompiler/Conversion/LLHToMath/LLHToMath.h"
#include "llcompiler/Conversion/LLHToHLO/LLHPreprocessingForHLO.h"
#include "llcompiler/Conversion/LLHToHLO/LLHToHLO.h"
#include "llcompiler/Conversion/LLHToShape/LLHToShape.h"
#include "llcompiler/Conversion/LLHToTensor/LLHToTensor.h"
#include "llcompiler/Conversion/LLHToTosa/LLHToTosa.h"
#include "llcompiler/Conversion/TosaToLinalgExtension/TosaToLinalgExtension.h"
#include "llcompiler/Conversion/StablehlotoLinalgExtension/StablehlotoLinalgExtension.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "llcompiler/Conversion/Passes.h.inc"

}  // namespace mlir

#endif  // INCLUDE_LLCOMPILER_CONVERSION_PASSES_H_
