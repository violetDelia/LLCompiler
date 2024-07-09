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
#include "onnx/shape_inference/implementation.h"

namespace llc::importer {
void OnnxImporter::init_form_json_(const mlir::StringRef &filename) {
  WARN(IMPORTER) << "never used json file initialize onnx model -> "
                 << filename.str();
  auto buf = mlir::openInputFile(filename);
  if (!buf) {
    ERROR(IMPORTER) << "open json file " << filename.str() << " failed!";
    return;
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
  auto status = google::protobuf::util::JsonStringToMessage(json, &model_);
  CHECK(IMPORTER, status.ok())
      << "convert json string to onnx::modelproto file faile!" << "\n\t"
      << filename.str() << " with error '" << status.ToString() + "'";
}

void OnnxImporter::init_form_onnx_(const mlir::StringRef &filename) {
  std::fstream input(filename.str(), std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    ERROR(IMPORTER) << "file " << filename.str() << " is opening!";
    return;
  }
  auto parse_success = model_.ParseFromIstream(&input);
  CHECK(IMPORTER, parse_success)
      << "onnx model parsing failed on " + filename.str();
}

void OnnxImporter::init_model_(const mlir::StringRef filename) {
  std::string error_msg;
  if (filename.endswith(".json")) {
    init_form_json_(filename);
  } else if (filename.endswith(".onnx")) {
    init_form_onnx_(filename);
  } else {
    FATAL(IMPORTER) << "unsupported file format!";
  }
  DEBUG(IMPORTER) << "onnx::ModelProto successfully initialized!";
}

OnnxImporter::OnnxImporter(const mlir::MLIRContext *context,
                           const OpBuilder *builder, const std::string path)
    : Importer(context, builder) {
  init_model_(path);
  // ONNX_NAMESPACE::LoadProtoFromPath<ONNX_NAMESPACE::ModelProto>(path,
  // model_); ONNX_NAMESPACE::shape_inference::InferShapes(model_);
  // std::unique_ptr<ONNX_NAMESPACE::Graph> g(
  //     ONNX_NAMESPACE::ImportModelProto(model_));
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
