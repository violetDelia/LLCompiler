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

#include <cstdint>
#include <iostream>
#include <string>

#include "google/protobuf/util/json_util.h"
#include "llcompiler/Importer/OnnxImporter.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/ir.h"
#include "onnx/onnx-ml.pb.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace llc::importer {
bool OnnxImporter::init_model_form_json_(const mlir::StringRef &filename,
                                         onnx::ModelProto *model) {
  WARN(IMPORTER) << "never used json file initialize onnx model -> "
                 << filename.str();
  auto buf = mlir::openInputFile(filename);
  if (!buf) {
    ERROR(IMPORTER) << "open json file " << filename.str() << " failed!";
    return false;
  }
  std::string json;
  for (llvm::line_iterator line(*buf, false), end; line != end; ++line) {
    if (line->ltrim(" \t").startswith("//")) continue;
    LOG(IMPORTER, line->contains("//"), 3)
        << "possible invalid end-of-line '//' comment in json input "
           "file "
        << filename.str() << ":" << line.line_number() << "\n";
    json.append(*line);
  }
  auto status = google::protobuf::util::JsonStringToMessage(json, model);
  CHECK(IMPORTER, status.ok())
      << "convert json string to onnx::modelproto file faile!" << "\n\t"
      << filename.str() << " with error '" << status.ToString() + "'";
  return status.ok();
}

bool OnnxImporter::init_model_form_onnx_(const mlir::StringRef &filename,
                                         onnx::ModelProto *model) {
  std::fstream input(filename.str(), std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    ERROR(IMPORTER) << "file " << filename.str() << " is opening!";
    return false;
  }
  auto parse_success = model->ParseFromIstream(&input);
  CHECK(IMPORTER, parse_success)
      << "onnx model parsing failed on " + filename.str();
  return parse_success;
}

bool OnnxImporter::init_model_(const mlir::StringRef filename,
                               onnx::ModelProto *model) {
  std::string error_msg;
  if (filename.endswith(".json")) {
    return init_model_form_json_(filename, model);
  } else if (filename.endswith(".onnx")) {
    return init_model_form_onnx_(filename, model);
  } else {
    FATAL(IMPORTER) << "unsupported file format!";
    return false;
  }
  DEBUG(IMPORTER) << "onnx::ModelProto successfully initialized!";
}

int64_t OnnxImporter::get_model_version_(const onnx::ModelProto &model) const {
  for (auto it = model.opset_import().begin(); it != model.opset_import().end();
       ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      auto version = it->version();
      INFO(IMPORTER) << "onnx version is: " << version;
      return version;
    }
  }
  ERROR(IMPORTER) << "can't find onnx version from onnx::ModelProto!";
  return -1;
}

bool OnnxImporter::check_model_legal_(const onnx::ModelProto &model) const {
  WARN_UNIMPLEMENTED(IMPORTER);
  return true;
}

onnx::ModelProto OnnxImporter::conver_model_version_to_(onnx::ModelProto *model,
                                                        const int64_t version) {
  return onnx::version_conversion::ConvertVersion(*model, version);
}

OnnxImporter::OnnxImporter(const mlir::MLIRContext *context,
                           const OpBuilder *builder,
                           const ImporterOption &option)
    : Importer(context, builder, option),
      convert_version_(option.onnx_convert_version) {
  if (option.importer_type == IMPORTER_TYPE::ONNX_FILE) {
    init_model_(option.filename, &model_);
  } else {
    UNIMPLEMENTED(IMPORTER) << " need support other importer types";
  }
  check_model_legal_(model_);
  auto onnx_version = get_model_version_(model_);
  if (onnx_version != convert_version_) {
    model_ = conver_model_version_to_(&model_, convert_version_);
    INFO(IMPORTER) << "convert onnx modle to version " << convert_version_;
  }
}

mlir::ModuleOp OnnxImporter::export_mlir_module() const {
  // mlir::ModuleOp Module = mlir::ModuleOp::create(builder_->getUnknownLoc());
  //  builder_->setInsertionPointToEnd(module_ getBody());
  //  auto func =
  //      m_builder.create<mlir::FuncOp>(m_builder.getUnknownLoc(), g->name(),
  //                                     get_func_type(g->inputs(),
  //                                     g->outputs()));
  //  mlir::Block *entryBlock = func.addEntryBlock();
  //  m_builder.setInsertionPointToStart(entryBlock);
  UNIMPLEMENTED(IMPORTER);
  return {};
}
};  // namespace llc::importer
