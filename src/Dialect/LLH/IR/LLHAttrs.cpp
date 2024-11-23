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

#include "Dialect/LLH/IR/LLHAttrs.h"

#include <cstddef>

#include "Dialect/LLH/IR/LLHEnums.h"
#include "Dialect/LLH/IR/LLHOps.h"
#include "Support/Logger.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#define GET_ATTRDEF_CLASSES
#include "Dialect/LLH/IR/LLHAttrs.cpp.inc"
namespace mlir::llh {

//===----------------------------------------------------------------------===//
// LLHDialect initialize method.
//===----------------------------------------------------------------------===//
void LLHDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/LLH/IR/LLHAttrs.cpp.inc"
      >();
}

size_t LayoutAttr::getFeatureIndex() {
  auto layout = getValue();
  switch (layout) {
    case Layout::NCHW:
      return 1;
    case Layout::NHWC:
      return 3;
  }
  UNIMPLEMENTED(llc::MLIR);
  return -1;
}

size_t LayoutAttr::getFirstSpatialIndex() {
  auto layout = getValue();
  switch (layout) {
    case Layout::NCHW:
      return 2;
    case Layout::NHWC:
      return 1;
  }
  UNIMPLEMENTED(llc::MLIR);
  return -1;
}

size_t LayoutAttr::getBatchIndex() {
  auto layout = getValue();
  switch (layout) {
    case Layout::NCHW:
      return 0;
    case Layout::NHWC:
      return 0;
  }
  UNIMPLEMENTED(llc::MLIR);
  return -1;
}

llvm::SmallVector<int64_t> LayoutAttr::addBatchAndFeature(
    llvm::ArrayRef<int64_t> kernel_shape) {
  llvm::SmallVector<int64_t> out;
  for (auto shape : kernel_shape) {
    out.push_back(shape);
  }
  if (getBatchIndex() < getFeatureIndex()) {
    out.insert(out.begin() + getBatchIndex(), 1);
    out.insert(out.begin() + getFeatureIndex(), 1);
  } else {
    out.insert(out.begin() + getFeatureIndex(), 1);
    out.insert(out.begin() + getBatchIndex(), 1);
  }
  return out;
}

}  // namespace mlir::llh
