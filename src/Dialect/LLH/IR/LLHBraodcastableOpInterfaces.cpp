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
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHEnums.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/SymbolInfer/Utils/SymbolAnalysis.h"
#include "llcompiler/Dialect/LLH/Utils/Utils.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Interfaces/BraodcastableOpInterfaces.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

// this feature is discarded
namespace mlir::llh {

#define RESHAPE_FOR_FUNCTION(OP) llvm::LogicalResult OP::reshapeAndBrodcast()

#define UNIMPLEMENTED_RESHAPE_FOR_FUNCTION(OP)                                \
  RESHAPE_FOR_FUNCTION(OP) {                                                  \
    WARN_UNIMPLEMENTED(llc::MLIR) << " op name:" << getOperationName().str(); \
    return llvm::failure();                                                   \
  }

#define SIMPLY_BINARY_ADD_BRAODCAST(OP)                                \
  RESHAPE_FOR_FUNCTION(OP) {                                           \
    auto op = getOperation();                                          \
    LLHPatternRewriter builder(op);                                    \
    if (checkBinaryNeedReshape(op).succeeded()) {                      \
      if (insertReshapeBeforeBinary<OP>(op, builder).failed())         \
        return llvm::failure();                                        \
    }                                                                  \
    if (checkBinaryNeedBroadcast(op).failed()) return llvm::failure(); \
    return insertBroadcastBeforeBinary<OP>(op, builder);               \
  }  // namespace mlir::llh

SIMPLY_BINARY_ADD_BRAODCAST(DivOp)
SIMPLY_BINARY_ADD_BRAODCAST(AddOp)
SIMPLY_BINARY_ADD_BRAODCAST(SubOp)
SIMPLY_BINARY_ADD_BRAODCAST(MulOp)
SIMPLY_BINARY_ADD_BRAODCAST(MaxOp)
SIMPLY_BINARY_ADD_BRAODCAST(MinOp)
SIMPLY_BINARY_ADD_BRAODCAST(CompareOp)

#undef RESHAPE_FOR_FUNCTION
#undef UNIMPLEMENTED_RESHAPE_FOR_FUNCTION
#undef SIMPLY_BINARY_ADD_BRAODCAST

}  // namespace mlir::llh
