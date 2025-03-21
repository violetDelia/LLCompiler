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
#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"

// using namespace ::mlir;

#include "llcompiler/Dialect/LLH/IR/LLHDialect.cpp.inc"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::llh {
//===----------------------------------------------------------------------===//
// LLHDialect InlinerInterface.
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with gpu
/// operations.
struct LLHInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// LLHDialect initialize method.
//===----------------------------------------------------------------------===//
void LLHDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "llcompiler/Dialect/LLH/IR/LLHOps.cpp.inc"
      >();
  registerTypes();
  registerAttributes();
  addInterfaces<LLHInlinerInterface>();
  addInterfaces<LLVMTranslationDialectInterface>();
}

}  // namespace mlir::llh
