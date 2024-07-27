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

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHTYPES_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHTYPES_H_

#include <atomic>
#include <cstdint>

#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"

namespace mlir::llh {
namespace detail {
struct TensorTypeStorage;
}  // namespace detail
enum SIGNED_TAG : uint32_t {
  SIGNLESS,
  SIGNED,
  UNSIGNED,
};

class DynamicDim {
  class InstanceID {
   public:
    InstanceID() = delete;
    static size_t Get() { return m_counter.fetch_add(1); }

   private:
    inline static std::atomic<uint64_t> m_counter = 0;
  };

 public:
  explicit DynamicDim(int64_t value) : value_(value), id_(InstanceID::Get()) {}
  explicit DynamicDim(int64_t value, bool is_dynamic)
      : value_(value), is_dynamic_(is_dynamic), id_(InstanceID::Get()) {}
  bool operator==(const DynamicDim &other) const {
    if (is_dynamic_) {
      return is_dynamic_ == other.is_dynamic_ && id_ == other.id_;
    }
    return is_dynamic_ == other.is_dynamic_ && value_ == other.value_;
  }
  int operator()() { return 0; }
  bool is_dynamic() const { return is_dynamic_; }
  int64_t value() const { return value_; }
  uint64_t id() const { return id_; }

 protected:
  int64_t value_ = -1;
  bool is_dynamic_ = false;
  uint64_t id_;
};
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const DynamicDim &dim) {
  if (dim.is_dynamic()) {
    os << dim.value();
  } else {
    os << "[";
    os << dim.id();
    os << "]";
  }
  return os;
}

}  // namespace mlir::llh

namespace mlir {
template <>
struct FieldParser<llh::DynamicDim> {
  static FailureOr<llh::DynamicDim> parse(AsmParser &parser) {
    UNIMPLEMENTED(::llc::IMPORTER);
    return mlir::llh::DynamicDim(1);
  }
};
}  // namespace mlir

#include "llcompiler/Dialect/LLH/IR/LLHEunms.h.inc"
#define GET_ATTRDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h.inc"

#endif  // INCLUDE_LLCOMPILER_DIALECT_LLH_IR_LLHTYPES_H_
