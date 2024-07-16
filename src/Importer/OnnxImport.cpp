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
#include <memory>
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
#include "onnx/version_converter/convert.h"

namespace llc::importer {
bool OnnxImporter::init_model_form_json_(const mlir::StringRef &filename,
                                         onnx::ModelProto *model) {
  WARN(IMPORTER) << "never used json file initialize onnx model -> "
                 << filename.str();
  auto buf = mlir::openInputFile(filename);
  if (!buf) {
    WRONG(IMPORTER) << "open json file " << filename.str() << " failed!";
    return false;
  }
  std::string json;
  for (llvm::line_iterator line(*buf, false), end; line != end; ++line) {
    if (line->ltrim(" \t").starts_with("//")) continue;
    LOG(IMPORTER, line->contains("//"), 3)
        << "possible invalid end-of-line '//' comment in json input "
           "file "
        << filename.str() << ":" << line.line_number() << "\n";
    json.append(*line);
  }
  auto status = google::protobuf::util::JsonStringToMessage(json, model);
  CHECK(IMPORTER, status.ok())
      << "convert json string to onnx::modelproto file faile!" << "\n\t"
      << filename.str() << " with WRONG '" << status.ToString() + "'";
  return status.ok();
}

bool OnnxImporter::init_model_form_onnx_(const mlir::StringRef &filename,
                                         onnx::ModelProto *model) {
  std::fstream input(filename.str(), std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    WRONG(IMPORTER) << "file " << filename.str() << " is opening!";
    return false;
  }
  auto parse_success = model->ParseFromIstream(&input);
  CHECK(IMPORTER, parse_success)
      << "onnx model parsing failed on " + filename.str();
  return parse_success;
}

bool OnnxImporter::init_model_(const mlir::StringRef filename,
                               onnx::ModelProto *model) {
  std::string WRONG_msg;
  if (filename.ends_with(".json")) {
    return init_model_form_json_(filename, model);
  } else if (filename.ends_with(".onnx")) {
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
  WRONG(IMPORTER) << "can't find onnx version from onnx::ModelProto!";
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

OnnxImporter::OnnxImporter(OpBuilder *builder, const ImporterOption &option)
    : Importer(builder, option), convert_version_(option.onnx_convert_version) {
  switch (option.importer_type) {
    case IMPORTER_TYPE::ONNX_FILE:
      init_model_(option.filename, &model_);
      break;
    default:
      FATAL(IMPORTER) << " need support other importer types";
  }
  check_model_legal_(model_);
  auto onnx_version = get_model_version_(model_);
  if (onnx_version != convert_version_) {
    model_ = conver_model_version_to_(&model_, convert_version_);
    INFO(IMPORTER) << "convert onnx modle to version " << convert_version_;
  }
  onnx::shape_inference::InferShapes(model_);
  INFO(IMPORTER) << "infer shapes of onnx model success!";
}

mlir::Type OnnxImporter::gen_type(mlir::OpBuilder *builder,
                                  const int32_t &elem_type) {
  DEBUG(IMPORTER) << "generating type: " << elem_type;
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return builder->getF32Type();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return builder->getF16Type();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return builder->getIntegerType(8, mlir::IntegerType::Unsigned);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      builder->getIntegerType(32, mlir::IntegerType::Signed);
    default:
      UNIMPLEMENTED(IMPORTER);
  }
  return mlir::Type();
}

mlir::ShapedType OnnxImporter::gen_type(mlir::OpBuilder *builder,
                                        onnx::Value const *value) {
  DEBUG(IMPORTER) << "generating shape type ";
  std::vector<int64_t> dims;
  for (auto dim : value->sizes()) {
    print_info << dim.dim;
    dims.emplace_back(dim.dim);
  }
  if (dims.size() > 0) {
    return mlir::RankedTensorType::get(dims,
                                       gen_type(builder, value->elemType()));
  } else {
    WARN(IMPORTER) << "Shape is unknown,  make 1 dim shape";
    return mlir::RankedTensorType::get({-1},
                                       gen_type(builder, value->elemType()));
  }
}

// mlir::ShapedType OnnxImporter::gen_type(mlir::OpBuilder *builder,
//                                         onnx::Value *value) {
//   UNIMPLEMENTED(IMPORTER);
// }

mlir::ModuleOp OnnxImporter::export_mlir_module() const {
  auto module =
      mlir::ModuleOp::create(builder_trace_.build().build().getUnknownLoc());
  auto graph = onnx::ImportModelProto(model_);
  builder_trace_.gen_mlir(&module, *graph);

  // auto func = build_->build().create<mlir::FuncOp>(
  //     m_builder.getUnknownLoc(), g->name(),
  //     get_func_type(g->inputs(), g->outputs()));

  // builder_->mlirGen(&module, model_);

  // module.push_back(mainFunc);
  // Create and set insertion point to entry block.
  // mainFunc.getBody().push_back(new mlir::Block);
  // builder_->builder_.setInsertionPointToStart(&mainFunc.getBody().back());
  // std::cout << std::addressof(*mainFunc.getOperation());
  module->dump();
  UNIMPLEMENTED(IMPORTER);
  return {};
}
};  // namespace llc::importer
