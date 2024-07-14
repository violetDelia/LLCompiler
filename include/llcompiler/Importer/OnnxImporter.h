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
#include <cstdint>

#include "llcompiler/Importer/Importer.h"
#include "llcompiler/Support/Core.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "onnx/common/file_utils.h"

#ifndef INCLUDE_LLCOMPILER_IMPORTER_ONNXIMPORTER_H_
#define INCLUDE_LLCOMPILER_IMPORTER_ONNXIMPORTER_H_

namespace llc::importer {
class OnnxImporter : public Importer {
 public:
  OnnxImporter(OpBuilder *builder, const ImporterOption &option);

  mlir::ModuleOp export_mlir_module() const final;

 protected:
  bool init_model_(const mlir::StringRef filename, onnx::ModelProto *model);
  bool init_model_form_json_(const mlir::StringRef &filename,
                             onnx::ModelProto *model);
  bool init_model_form_onnx_(const mlir::StringRef &filename,
                             onnx::ModelProto *model);
  bool check_model_legal_(const onnx::ModelProto &model) const;
  int64_t get_model_version_(const onnx::ModelProto &model) const;
  onnx::ModelProto conver_model_version_to_(onnx::ModelProto *model,
                                            const int64_t version);

  onnx::ModelProto model_;
  int64_t convert_version_ = 15;
};
}  // namespace llc::importer
#endif  // INCLUDE_LLCOMPILER_IMPORTER_ONNXIMPORTER_H_
