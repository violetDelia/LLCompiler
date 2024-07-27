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
#ifndef INCLUDE_LLCOMPILER_FRONTEND_ONNX_ONNXIMPORTER_H_
#define INCLUDE_LLCOMPILER_FRONTEND_ONNX_ONNXIMPORTER_H_
#include <cstdint>

#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Frontend/Core/Importer.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"


namespace llc::front {
class OnnxImporter : public Importer {
 public:
  OnnxImporter(mlir::MLIRContext *context, const ImporterOption &option);

  mlir::ModuleOp export_mlir_module() const final;

 protected:
  bool init_model_(const mlir::StringRef filename,
                   ONNX_NAMESPACE::ModelProto *model);
  bool init_model_form_json_(const mlir::StringRef &filename,
                             ONNX_NAMESPACE::ModelProto *model);
  bool init_model_form_onnx_(const mlir::StringRef &filename,
                             ONNX_NAMESPACE::ModelProto *model);
  bool check_model_legal_(const ONNX_NAMESPACE::ModelProto &model) const;
  int64_t get_model_version_(const ONNX_NAMESPACE::ModelProto &model) const;
  ONNX_NAMESPACE::ModelProto conver_model_version_to_(
      ONNX_NAMESPACE::ModelProto *model, const int64_t version);

  ONNX_NAMESPACE::ModelProto model_;
  int64_t onnx_version_ = 22;
};
}  // namespace llc::front
#endif  // INCLUDE_LLCOMPILER_FRONTEND_ONNX_ONNXIMPORTER_H_
