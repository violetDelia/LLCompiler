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
/**
 * @file Importer.h
 * @brief Importer can convert input to mlir::ModuleOp,this is a interface
 * class.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */
#include <iterator>
#include <string>

#include "llcompiler/Importer/OpBuilder.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#ifndef INCLUDE_LLCOMPILER_IMPORTER_IMPORTER_H_
#define INCLUDE_LLCOMPILER_IMPORTER_IMPORTER_H_
#define IMPORTER_GEN_TYPE(Type, ...) \
  static Type gen_type(mlir::OpBuilder *builder, __VA_ARGS__);

namespace llc::importer {
struct ImporterOption {
  std::string filename;
  uint64_t onnx_convert_version;
  IMPORTER_TYPE importer_type;
  TARGET_DIALECT target_dialect;
};

class Importer {
 public:
  Importer(OpBuilder *builder, const ImporterOption &option);
  virtual ~Importer();
  virtual mlir::ModuleOp export_mlir_module() const = 0;

  mlir::TypeRange gen_types(
      mlir::OpBuilder *builder,
      const onnx::ArrayRef<const onnx::Value *> &container);

 protected:
  OpBuilderTrace builder_trace_;
};

template <class Importer, class Container>
mlir::TypeRange gen_types(mlir::OpBuilder *builder,
                          const Container &container) {
  llvm::SmallVector<mlir::Type, 4> types;
  for (auto value : container) {
    types.push_back(Importer::gen_type(builder, value));
  }
  return types;
}
}  // namespace llc::importer

#endif  // INCLUDE_LLCOMPILER_IMPORTER_IMPORTER_H_
