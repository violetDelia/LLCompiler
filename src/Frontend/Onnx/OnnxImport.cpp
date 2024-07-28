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
#include <cstring>
#include <fstream>
#include <ios>
#include <map>
#include <memory>
#include <string>

#include "google/protobuf/util/json_util.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Frontend/Core/Builder.h"
#include "llcompiler/Frontend/Core/Importer.h"
#include "llcompiler/Frontend/Onnx/OnnxImporter.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "onnx/common/array_ref.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/tensor.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace llc::front {
bool OnnxImporter::init_model_form_json_(const mlir::StringRef &filename,
                                         ONNX_NAMESPACE::ModelProto *model) {
  WARN(IMPORTER) << "never used json file initialize onnx model -> "
                 << filename.str();
  std::unique_ptr<llvm::MemoryBuffer> buf = mlir::openInputFile(filename);
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
  if (filename.ends_with(".json")) {
    WARN(IMPORTER) << "json";
    // return init_model_form_json_(filename, model);
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
  auto op_set = model.opset_import();
  for (auto it = op_set.begin(); it != op_set.end(); ++it) {
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

OnnxImporter::OnnxImporter(Builder *builder, const ImporterOption &option)
    : Importer(builder, option) {
  switch (option.frontend_type) {
    case FRONTEND_TYPE::ONNX_FILE:
      init_model_(option.filename, &model_);
      break;
    default:
      FATAL(IMPORTER) << " need support other importer types";
  }
  check_model_legal_(model_);
  auto onnx_version = get_model_version_(model_);
  if (option.onnx_convert && onnx_version != option.onnx_convert_version) {
    WARN(IMPORTER)
        << "convert onnx modle to version " << option.onnx_convert_version
        << ", convert onnx modle may raise unexpected error at "
           "ONNX_NAMESPACE::ConvertVersion, it is recommended use onnx file "
           "of version 22.";
    model_ = conver_model_version_to_(&model_, option.onnx_convert_version);
  }
  ONNX_NAMESPACE::shape_inference::InferShapes(model_);
  INFO(IMPORTER) << "infer shapes of onnx model success!";
}

mlir::Type OnnxImporter::mlir_gen(mlir::OpBuilder *builder,
                                  const int32_t &elem_type) const {
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

mlir::ShapedType OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Value &value) const {
  llvm::SmallVector<int64_t> dims;
  for (auto dim : value.sizes()) {
    if (dim.dim < 0) {
      dims.emplace_back(mlir::ShapedType::kDynamic);
    } else {
      dims.emplace_back(dim.dim);
    }
  }
  auto type = mlir_gen(builder, value.elemType());
  return mlir::RankedTensorType::get(dims, type);
}

llvm::SmallVector<mlir::Type> OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder,
    const ONNX_NAMESPACE::ArrayRef<const ONNX_NAMESPACE::Value *> &values)
    const {
  llvm::SmallVector<mlir::Type> types;
  for (auto &value : values) {
    types.push_back(mlir_gen(builder, *value));
  }
  return types;
}

#define CREATE_WEIGHT_OP_FROM_ONNX(Onnx_Type, Type, weight, shape, builder)  \
  if (weight.elem_type() ==                                                  \
      ONNX_NAMESPACE::TensorProto_DataType_##Onnx_Type) {                    \
    auto element_size = get_element_size_form(shape);                        \
    auto data_begin = weight.data<Type>();                                   \
    auto value = mlir::DenseElementsAttr::get(                               \
        shape, llvm::ArrayRef<Type>(data_begin, data_begin + element_size)); \
    return build_op<mlir::tosa::ConstOp>(builder, shape, value);             \
  }

mlir::tosa::ConstOp OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Tensor &weight,
    std::map<std::string, mlir::ShapedType> *weight_shape_map) const {
  auto name = weight.name();
  auto shape = weight_shape_map->at(name);
  CHECK(IMPORTER, shape.hasStaticShape()) << "it not static shape for weight ";
  auto elem_type = shape.getElementType();
  auto data = weight.data<int>();
  if (elem_type != mlir_gen(builder, weight.elem_type())) {
    WARN(IMPORTER) << "element_type of initializer is conflict!";
  }
  CREATE_WEIGHT_OP_FROM_ONNX(FLOAT, float, weight, shape, builder)
  CREATE_WEIGHT_OP_FROM_ONNX(DOUBLE, double, weight, shape, builder)
  CREATE_WEIGHT_OP_FROM_ONNX(INT32, int32_t, weight, shape, builder)
  CREATE_WEIGHT_OP_FROM_ONNX(INT64, int64_t, weight, shape, builder)
  CREATE_WEIGHT_OP_FROM_ONNX(UINT64, uint64_t, weight, shape, builder)
  UNIMPLEMENTED(IMPORTER) << "unimplemented weight data type: "
                          << weight.elem_type();
}

#undef CREATE_WEIGHT_OP_FROM_ONNX
#define IF_NODE_NAME_IS(name) if (!strcmp(node.kind().toString(), name))

mlir::Operation *OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Node &node,
    std::map<std::string, mlir::Value *> *value_map) const {
  IF_NODE_NAME_IS("Undefined") { return nullptr; }
  IF_NODE_NAME_IS("Conv") {
    auto inputs = node.inputs();
    auto outputs = node.outputs();
    auto input_size = inputs.size();
    auto output = mlir_gen(builder, outputs[0]);
    auto input = value_map->at(inputs[0]->uniqueName());
    auto weight = value_map->at(inputs[1]->uniqueName());
    auto pad = node.is(ONNX_NAMESPACE::BuiltinSymbol::kpad);
    auto padattr = builder->getDenseI64ArrayAttr(pad);
    padattr.dump();
    auto stride = node.is(ONNX_NAMESPACE::BuiltinSymbol::kstride);
    auto strideattr = builder->getDenseI64ArrayAttr(stride);
    strideattr.dump();
    auto dilation = node.is(ONNX_NAMESPACE::BuiltinSymbol::kdilation);
    auto dilationattr = builder->getDenseI64ArrayAttr(dilation);
    dilationattr.dump();
  }
  UNIMPLEMENTED(IMPORTER) << "unimplemented op generate: "
                          << node.kind().toString();
  return build_op<mlir::llh::UndefinedOp>(builder, node.kind().toString());
}

// #undef IF_NODE_NAME_IS
mlir::func::FuncOp OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Graph &graph) const {
  INFO(IMPORTER) << "----- building func op-----";
  auto inputs = graph.inputs();
  auto func_inputs = mlir_gen(builder, graph.inputs());
  auto func_outputs = mlir_gen(builder, graph.outputs());
  auto func_type = builder->getFunctionType(func_inputs, func_outputs);
  auto func = build_op<mlir::func::FuncOp>(builder, graph.name(), func_type);
  std::map<std::string, mlir::Value *> value_map;
  auto block = func.addEntryBlock();
  auto inputs_size = inputs.size();
  for (int i = 0; i < inputs_size; ++i) {
    auto name = inputs[i]->uniqueName();
    auto value = block->getArgument(i);
    value_map[name] = &value;
  }
  std::set<std::string> weight_set;
  for (auto weight_name : graph.initializer_names()) {
    weight_set.insert(weight_name);
  }
  std::map<std::string, mlir::ShapedType> weight_shape_map;
  for (auto node : graph.nodes()) {
    for (auto &input : node->inputs()) {
      auto name = input->uniqueName();
      if (weight_set.find(name) == weight_set.end()) continue;
      weight_shape_map[name] = mlir_gen(builder, *input);
    }
  }
  INFO(IMPORTER) << "----- building weight ops -----";
  for (auto &weight : graph.initializers()) {
    auto weight_op = mlir_gen(builder, weight, &weight_shape_map);
    add_attr(weight_op, LLCOperationNmaeAttr, weight.name());
    block->push_back(weight_op);
    auto value = weight_op.getResult();
    value_map[weight.name()] = &value;
  }
  INFO(IMPORTER) << "----- building node ops -----";
  for (auto node : graph.nodes()) {
    auto op = mlir_gen(builder, *node, &value_map);
    if (!op) continue;
    add_attr(op, LLCOperationNmaeAttr, node->name());
    block->push_back(op);
  }

  return func;
}

mlir::ModuleOp OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::ModelProto &model) const {
  INFO(IMPORTER) << "----- building module op -----";
  auto module = build_op<mlir::ModuleOp>(builder);
  auto graph = ONNX_NAMESPACE::ImportModelProto(model);
  auto func = mlir_gen(builder, *graph);
  module.push_back(func);
  return module;
}

mlir::ModuleOp OnnxImporter::export_mlir_module() const {
  auto builder = builder_->builder();
  auto module = mlir_gen(&builder, model_);
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
};  // namespace llc::front
