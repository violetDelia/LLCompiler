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
 * @file OnnxImporter.h
 * @brief implementation of Importer that convert onnx.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */
#include <string>

#include "llcompiler/Importer/Importer.h"
#include "llcompiler/Support/Core.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ir/OwningOpRef.h"
#include "onnx/common/file_utils.h"
#include "onnx/onnx_pb.h"

#ifndef INCLUDE_LLCOMPILER_IMPORTER_ONNXIMPORTER_H_
#define INCLUDE_LLCOMPILER_IMPORTER_ONNXIMPORTER_H_

namespace llc::importer {
class OnnxImporter : public Importer {
 public:
  OnnxImporter(const mlir::MLIRContext *context, const OpBuilder *builder,
               const std::string path);

  OnnxImporter(const mlir::MLIRContext *context, const OpBuilder *builder,
               const ONNX_NAMESPACE::ModelProto model);

  mlir::ModuleOp export_mlir_module() const final;

 protected:
  void init_model_(const mlir::StringRef filename);
  void init_form_json_(const mlir::StringRef &filename);
  void init_form_onnx_(const mlir::StringRef &filename);

  onnx::ModelProto model_;
};
}  // namespace llc::importer
#endif  // INCLUDE_LLCOMPILER_IMPORTER_ONNXIMPORTER_H_
