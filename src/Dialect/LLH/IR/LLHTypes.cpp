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
#include <utility>

#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::llh {

namespace detail {
//===----------------------------------------------------------------------===//
// TensorTypeStorage
//===----------------------------------------------------------------------===//
struct TensorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy =
      std::tuple<::llvm::ArrayRef<::mlir::llh::DynamicDim>, ::mlir::Type>;
  TensorTypeStorage(::llvm::ArrayRef<::mlir::llh::DynamicDim> dims,
                    ::mlir::Type type)
      : dims(std::move(dims)), type(std::move(type)) {
  }

  static ::llvm::ArrayRef<::mlir::llh::DynamicDim *> new_dims(
      ::llvm::ArrayRef<::mlir::llh::DynamicDim> dims) {
    llvm::SmallVector<::mlir::llh::DynamicDim *, 4> new_dims;
    for (auto dim : dims) {
      new_dims.push_back(&dim);
    }
    return new_dims;
  }

  KeyTy getAsKey() const {
    return KeyTy(dims, type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (new_dims(dims) == new_dims(std::get<0>(tblgenKey))) &&
           (type == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(new_dims(std::get<0>(tblgenKey)),
                                std::get<1>(tblgenKey));
  }

  static TensorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator,
                                      KeyTy &&tblgenKey) {
    auto dims = std::move(std::get<0>(tblgenKey));
    auto type = std::move(std::get<1>(tblgenKey));
    dims = allocator.copyInto(dims);

    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::move(dims), std::move(type));
  }

  ::llvm::ArrayRef<::mlir::llh::DynamicDim> dims;
  ::mlir::Type type;
};
}  // namespace detail
//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//
::llvm::ArrayRef<::mlir::llh::DynamicDim> TensorType::getDims() const {
  getImpl()->dims;
}
::mlir::Type TensorType::getType() const { getImpl()->type; }
//===----------------------------------------------------------------------===//
// LLHDialect initialize method.
//===----------------------------------------------------------------------===//
void LLHDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llcompiler/Dialect/LLH/IR/LLHTypes.cpp.inc"
      >();
}

void printSIGNED_TAG(::mlir::AsmPrinter &printer, SIGNED_TAG tag) {
  switch (tag) {
    case UNSIGNED:
      printer << "u";
    default:
      return;
  }
}

llvm::ParseResult parseSIGNED_TAG(::mlir::AsmParser &parser, SIGNED_TAG &tag) {
  WARN_UNIMPLEMENTED(llc::UTILITY);
  return mlir::success();
}

void printDynamicDim(::mlir::AsmPrinter &printer,
                     ::llvm::ArrayRef<::mlir::llh::DynamicDim> dim) {
  printer << "u";
}

llvm::ParseResult parseDynamicDim(
    ::mlir::AsmParser &parser,
    llvm::SmallVector<::mlir::llh::DynamicDim> &dim) {
  WARN_UNIMPLEMENTED(llc::UTILITY);
  return mlir::success();
}
//===----------------------------------------------------------------------===//
// LLH type verify
//===----------------------------------------------------------------------===//
::llvm::LogicalResult IntType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned width, SIGNED_TAG signed_tag) {
  if (width > Max_Width) {
    return emitError() << "IntType max bitwidth cant greater than "
                       << Max_Width;
  }
  return llvm::success();
}

::llvm::LogicalResult TensorType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()>,
    llvm::ArrayRef<::mlir::llh::DynamicDim>, mlir::Type) {
  return llvm::success();
}
}  // namespace mlir::llh

#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHTypes.cpp.inc"
