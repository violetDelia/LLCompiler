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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "onnx/common/file_utils.h"
#include "onnx/common/ir.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace llc::importer {
bool OnnxImporter::init_model_form_json_(const mlir::StringRef &filename,
                                         ONNX_NAMESPACE::ModelProto *model) {
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
      << "convert json string to ONNX_NAMESPACE::modelproto file faile!"
      << "\n\t" << filename.str() << " with WRONG '" << status.ToString() + "'";
  return status.ok();
}

bool OnnxImporter::init_model_form_onnx_(const mlir::StringRef &filename,
                                         ONNX_NAMESPACE::ModelProto *model) {
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
                               ONNX_NAMESPACE::ModelProto *model) {
  std::string WRONG_msg;
  if (filename.ends_with(".json")) {
    return init_model_form_json_(filename, model);
  } else if (filename.ends_with(".onnx")) {
    return init_model_form_onnx_(filename, model);
  } else {
    FATAL(IMPORTER) << "unsupported file format!";
    return false;
  }
  DEBUG(IMPORTER) << "ONNX_NAMESPACE::ModelProto successfully initialized!";
}

int64_t OnnxImporter::get_model_version_(
    const ONNX_NAMESPACE::ModelProto &model) const {
  for (auto it = model.opset_import().begin(); it != model.opset_import().end();
       ++it) {
    if (it->domain() == "" || it->domain() == "ai.onnx") {
      auto version = it->version();
      INFO(IMPORTER) << "onnx version is: " << version;
      if (version != ONNX_ADAPTED_VERSION) {
        WARN(IMPORTER)
            << "onnx version is not 22, if an exception occurs during cannot "
               "import, please use onnx file of version 22.";
      }
      return version;
    }
  }
  WRONG(IMPORTER) << "can't find onnx version from ONNX_NAMESPACE::ModelProto!";
  return -1;
}

bool OnnxImporter::check_model_legal_(
    const ONNX_NAMESPACE::ModelProto &model) const {
  WARN_UNIMPLEMENTED(IMPORTER);
  return true;
}

ONNX_NAMESPACE::ModelProto OnnxImporter::conver_model_version_to_(
    ONNX_NAMESPACE::ModelProto *model, const int64_t version) {
  return ONNX_NAMESPACE::version_conversion::ConvertVersion(*model, version);
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
  if (option.onnx_convert && onnx_version != convert_version_) {
    WARN(IMPORTER)
        << "convert onnx modle to version " << convert_version_
        << ", convert onnx modle may raise unexpected error at "
           "ONNX_NAMESPACE::ConvertVersion, it is recommended use onnx file "
           "of version 22.";
    model_ = conver_model_version_to_(&model_, convert_version_);
  }
  ONNX_NAMESPACE::shape_inference::InferShapes(model_);
  INFO(IMPORTER) << "infer shapes of onnx model success!";
}

IMPORTER_GEN_TYPE_IMPL(OnnxImporter, mlir::Type, const int32_t &elem_type) {
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return builder->getF32Type();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return builder->getF16Type();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return builder->getIntegerType(8, mlir::IntegerType::Unsigned);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return builder->getIntegerType(32, mlir::IntegerType::Signed);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return builder->getIntegerType(64, mlir::IntegerType::Signed);
    default:
      UNIMPLEMENTED(IMPORTER) << "  onnx element type is " << elem_type << "!";
  }
  return mlir::Type();
}

IMPORTER_GEN_TYPE_IMPL(OnnxImporter, mlir::ShapedType,
                       const ONNX_NAMESPACE::Value *value) {
  llvm::SmallVector<int64_t> dims;
  for (auto dim : value->sizes()) {
    if (dim.dim < 0) {
      dims.emplace_back(mlir::ShapedType::kDynamic);
    } else {
      dims.emplace_back(dim.dim);
    }
  }
  return mlir::RankedTensorType::get(dims,
                                     gen_type(builder, value->elemType()));
}

mlir::ModuleOp OnnxImporter::export_mlir_module() const {
  auto module =
      mlir::ModuleOp::create(builder_trace_.build().build().getUnknownLoc());
  auto graph = ONNX_NAMESPACE::ImportModelProto(model_);
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
