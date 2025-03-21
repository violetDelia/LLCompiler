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
#include "llcompiler/Dialect/IRExtension/IR/Attrs.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/Utility/Attribute.h"
#include "llcompiler/Dialect/Utility/Builder.h"
#include "llcompiler/Support/Macro.h"
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
#include "llvm/Support/WithColor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "onnx/common/array_ref.h"
#include "onnx/common/interned_strings.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/tensor.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace llc::front {
namespace helper {

// enum class AttributeKind : uint8_t {
//   f,
//   fs,
//   i,
//   is,
//   s,
//   ss,
//   t,
//   ts,
//   g,
//   gs,
//   tp,
//   tps
// };

void info_node_attrs(const ONNX_NAMESPACE::Node &node) {
  for (auto s : node.attributeNames()) {
    INFO(IMPORTER) << s.toString();
    INFO(IMPORTER) << static_cast<int>(node.kindOf(s));
  }
}
}  // namespace helper

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

OnnxImporter::OnnxImporter(Builder *builder, const FrontEndOption &option)
    : Importer(builder, option) {
  switch (option.frontend_type) {
    case FRONTEND_TYPE::ONNX_FILE:
      init_model_(option.input_file, &model_);
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
  ONNX_NAMESPACE::shape_inference::InferShapes(
      model_, ONNX_NAMESPACE::OpSchemaRegistry::Instance(),
      ONNX_NAMESPACE::ShapeInferenceOptions(true, 0, true));
  INFO(IMPORTER) << "infer shapes of onnx model success!";
}

mlir::Type OnnxImporter::mlir_gen(mlir::OpBuilder *builder,
                                  const int32_t &elem_type) const {
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      WARN(IMPORTER) << "undefine data type,default use f32.";
      return builder->getF32Type();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return builder->getF32Type();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return builder->getF16Type();
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return builder->getIntegerType(8, mlir::IntegerType::Unsigned);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return builder->getI32Type();
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return builder->getI64Type();
    default:
      UNIMPLEMENTED(IMPORTER) << "  onnx element type is " << elem_type << "!";
  }
  return mlir::Type();
}

mlir::RankedTensorType OnnxImporter::mlir_gen(
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
  auto context = builder->getContext();
  // auto encode = mlir::ex::EncodingAttr::get(context, mlir::ex::Layout::NCHW);
  auto tensor = mlir::RankedTensorType::get(dims, type);
  return tensor;
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

#define MLIR_GEN_TENSOR_CONST_OP(Onnx_Type, Type, weight, tensor, builder, Op) \
  if (weight.elem_type() ==                                                    \
      ONNX_NAMESPACE::TensorProto_DataType_##Onnx_Type) {                      \
    auto element_size = getElementSizeFrom(tensor);                            \
    auto data_begin = weight.data<Type>();                                     \
    auto value_date =                                                          \
        llvm::ArrayRef<Type>(data_begin, data_begin + element_size);           \
    auto value = mlir::DenseElementsAttr::get(tensor, value_date);             \
    auto op = build_op<Op>(builder, tensor, value);                            \
    DEBUG_BUILDED_OP(IMPORTER, op);                                            \
    return op;                                                                 \
  }

mlir::Operation *OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Tensor &weight,
    std::map<std::string, mlir::RankedTensorType> *tensor_map) const {
  auto name = weight.name();
  auto tensor = tensor_map->at(name);
  CHECK(IMPORTER, tensor.hasStaticShape()) << "it not static shape for weight ";
  auto elem_type = tensor.getElementType();
  if (elem_type != mlir_gen(builder, weight.elem_type())) {
    WARN(IMPORTER) << "element_type of initializer is conflict!";
  }
  MLIR_GEN_TENSOR_CONST_OP(FLOAT, float, weight, tensor, builder,
                           mlir::llh::WeightOp)
  MLIR_GEN_TENSOR_CONST_OP(DOUBLE, double, weight, tensor, builder,
                           mlir::llh::WeightOp)
  MLIR_GEN_TENSOR_CONST_OP(INT32, int32_t, weight, tensor, builder,
                           mlir::llh::WeightOp)
  MLIR_GEN_TENSOR_CONST_OP(INT64, int64_t, weight, tensor, builder,
                           mlir::llh::ConstantOp)
  MLIR_GEN_TENSOR_CONST_OP(UINT64, uint64_t, weight, tensor, builder,
                           mlir::llh::ConstantOp)
  UNIMPLEMENTED(IMPORTER) << "unimplemented weight data type: "
                          << weight.elem_type();
}
#undef MLIR_GEN_TENSOR_CONST_OP

#define INTS_ATTR(key, builder, node)     \
  case ONNX_NAMESPACE::AttributeKind::is: \
    return builder->getDenseI64ArrayAttr(node.is(key));

#define INT_ATTR(key, builder, node)     \
  case ONNX_NAMESPACE::AttributeKind::i: \
    return builder->getI64IntegerAttr(node.i(key));

#define STRING_ATTR(key, builder, node)  \
  case ONNX_NAMESPACE::AttributeKind::s: \
    return builder->getStringAttr(node.s(key));

mlir::Attribute OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Node &node,
    const ONNX_NAMESPACE::BuiltinSymbol &attr_key) const {
  auto attr_kind = node.kindOf(attr_key);
  switch (attr_kind) {
    INT_ATTR(attr_key, builder, node)
    INTS_ATTR(attr_key, builder, node)
    STRING_ATTR(attr_key, builder, node)
  }
  UNIMPLEMENTED(IMPORTER) << "unknown attribute: "
                          << ONNX_NAMESPACE::Symbol(attr_key).toString()
                          << ". kind of attribute is ["
                          << static_cast<int>(node.kindOf(attr_key)) << "].";
}

#undef INTS_ATTR
#undef INT_ATTR
#undef STRING_ATTR

#define IF_NODE_NAME_IS(name) if (!strcmp(node.kind().toString(), name))
#define GET_ATTR(mlir_attr_name, key_name, builder, node)           \
  auto key_name##_key = ONNX_NAMESPACE::BuiltinSymbol::k##key_name; \
  auto key_name##_attr = mlir_gen(builder, node, key_name##_key);   \
  auto mlir_attr_name = builder->getNamedAttr(#mlir_attr_name, key_name##_attr);
#define ADD_ATTR(op, mlir_attr_name, key_name, builder, node)       \
  auto key_name##_key = ONNX_NAMESPACE::BuiltinSymbol::k##key_name; \
  auto key_name##_attr = mlir_gen(builder, node, key_name##_key);   \
  op.getOperation()->setAttr(mlir_attr_name, key_name##_attr);

#define BUILD_UNARY_OP(op_name, OP, input_index, output_index)    \
  auto inputs = node.inputs();                                    \
  auto outputs = node.outputs();                                  \
  auto output = mlir_gen(builder, *outputs[output_index]);        \
  auto input1 = value_map->at(inputs[input_index]->uniqueName()); \
  auto loc = builder->getUnknownLoc();                            \
  auto out_types = ::mlir::TypeRange{output};                     \
  auto operands = ::mlir::ValueRange{input1};                     \
  auto op_name = build_op<OP>(builder, out_types, operands);      \
  DEBUG_BUILDED_OP(IMPORTER, op_name)

#define COMMON_UNARY_OP(name, OP, input_index, output_index) \
  IF_NODE_NAME_IS(name) {                                    \
    BUILD_UNARY_OP(op_name, OP, input_index, output_index)   \
    return op_name;                                          \
  }

#define BUILD_BINARY_OP(op_name, OP, input1_index, input2_index, output_index) \
  auto inputs = node.inputs();                                                 \
  auto outputs = node.outputs();                                               \
  auto input_size = inputs.size();                                             \
  auto output = mlir_gen(builder, *outputs[output_index]);                     \
  auto input1 = value_map->at(inputs[input1_index]->uniqueName());             \
  auto input2 = value_map->at(inputs[input2_index]->uniqueName());             \
  auto loc = builder->getUnknownLoc();                                         \
  auto out_types = ::mlir::TypeRange{output};                                  \
  auto operands = ::mlir::ValueRange{input1, input2};                          \
  auto op_name = build_op<OP>(builder, out_types, operands);                   \
  DEBUG_BUILDED_OP(IMPORTER, op_name)

#define COMMON_BINARY_OP(name, OP, input1_index, input2_index, output_index) \
  IF_NODE_NAME_IS(name) {                                                    \
    BUILD_BINARY_OP(op_name, OP, input1_index, input2_index, output_index)   \
    return op_name;                                                          \
  }

mlir::Operation *OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Node &node,
    std::map<std::string, mlir::Value> *value_map) const {
  IF_NODE_NAME_IS("Undefined") { return nullptr; }
  COMMON_UNARY_OP("Relu", mlir::llh::ReluOp, 0, 0)
  COMMON_BINARY_OP("Add", mlir::tosa::AddOp, 0, 1, 0)
  COMMON_BINARY_OP("MatMul", mlir::llh::MatMulOp, 0, 1, 0)
  IF_NODE_NAME_IS("Reshape") {
    BUILD_UNARY_OP(op, mlir::tosa::ReshapeOp, 0, 0)
    op.setNewShape(output.getShape());
    return op;
  }
  IF_NODE_NAME_IS("MaxPool") {
    BUILD_UNARY_OP(op, mlir::tosa::MaxPool2dOp, 0, 0)
    ADD_ATTR(op, "kernel", kernel_shape, builder, node)
    ADD_ATTR(op, "stride", strides, builder, node)
    ADD_ATTR(op, "pad", pads, builder, node)
    return op;
  }
  IF_NODE_NAME_IS("Conv") {
    auto inputs = node.inputs();
    auto outputs = node.outputs();
    auto input_size = inputs.size();
    auto output = mlir_gen(builder, *outputs[0]);
    auto input = value_map->at(inputs[0]->uniqueName());
    auto weight = value_map->at(inputs[1]->uniqueName());
    mlir::Value blas;
    if (input_size == 3) {
      blas = value_map->at(inputs[2]->uniqueName());
    } else if (input_size == 2) {
      auto m = getShapeFrom(weight.getType())[0];
      mlir::SmallVector<double> values(m, 0);
      auto ele_type = output.getElementType();
      auto const_op = create_tosa_const(builder, {m}, values, ele_type,
                                        builder->getUnknownLoc());
      input.getParentBlock()->push_back(const_op);
      blas = const_op;
    } else {
      FATAL(IMPORTER) << "ERROR ONNX IR!";
    }
    auto out_types = mlir::TypeRange{output};
    auto operands = mlir::ValueRange{input, weight, blas};
    auto op = build_op<mlir::llh::ConvOp>(builder, out_types, operands);
    ADD_ATTR(op, "stride", strides, builder, node)
    ADD_ATTR(op, "dilation", dilations, builder, node)
    ADD_ATTR(op, GroupAttr, group, builder, node)
    ADD_ATTR(op, KernelShapeAttr, kernel_shape, builder, node)
    auto in_shape =
        mlir::cast<mlir::RankedTensorType>(input.getType()).getShape();
    auto stride = op.getStride();
    auto out_shape = output.getShape();
    auto dilation = op.getDilation();
    auto kernel_shape =
        mlir::cast<mlir::DenseI64ArrayAttr>(kernel_shape_attr).asArrayRef();
    auto pads = mlir::SmallVector<int64_t>();
    for (int i = 0; i < in_shape.size() - 2; i++) {
      auto pad = ((out_shape[i + 2] - 1) * stride[i] + 1 - in_shape[i + 2] +
                  dilation[i] * (kernel_shape[i] - 1)) /
                 2;
      pads.push_back(pad);
      pads.push_back(pad);
    }
    op.setPad(pads);
    DEBUG_BUILDED_OP(IMPORTER, op)
    return op;
  }
  UNIMPLEMENTED(IMPORTER) << "unimplemented op generate: "
                          << node.kind().toString();
  return build_op<mlir::llh::UndefinedOp>(builder, node.kind().toString());
}
#undef IF_NODE_NAME_IS
#undef GET_ATTR
#undef BUILD_UNARY_OP
#undef COMMON_UNARY_OP
#undef BUILD_BINARY_OP
#undef COMMON_BINARY_OP
#undef ADD_ATTR

mlir::func::FuncOp OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::Graph &graph) const {
  INFO(IMPORTER) << "----- building func op-----";
  auto loc = builder->getUnknownLoc();
  auto inputs = graph.inputs();
  auto outputs = graph.outputs();
  auto func_inputs = mlir_gen(builder, graph.inputs());
  auto func_outputs = mlir_gen(builder, graph.outputs());
  auto func_type = builder->getFunctionType(func_inputs, func_outputs);
  auto func = builder->create<mlir::func::FuncOp>(loc, "main", func_type);
  DEBUG_BUILDED_OP(IMPORTER, func);
  auto block = func.addEntryBlock();
  std::map<std::string, mlir::Value> value_map;
  auto inputs_size = inputs.size();
  for (int i = 0; i < inputs_size; ++i) {
    auto name = inputs[i]->uniqueName();
    value_map[name] = block->getArgument(i);
  }
  std::set<std::string> weight_set;
  for (auto weight_name : graph.initializer_names()) {
    weight_set.insert(weight_name);
  }
  std::map<std::string, mlir::RankedTensorType> weight_shape_map;
  for (auto node : graph.nodes()) {
    for (auto &input : node->inputs()) {
      auto name = input->uniqueName();
      if (weight_set.find(name) == weight_set.end()) continue;
      weight_shape_map[name] = mlir_gen(builder, *input);
    }
  }
  INFO(IMPORTER) << "----- building weight ops -----";
  for (auto &tensor : graph.initializers()) {
    auto tensor_op = mlir_gen(builder, tensor, &weight_shape_map);
    value_map[tensor.name()] = tensor_op->getResult(0);
    block->push_back(tensor_op);
  }
  INFO(IMPORTER) << "----- building node ops -----";
  for (auto node : graph.nodes()) {
    auto op = mlir_gen(builder, *node, &value_map);
    if (!op) continue;
    auto outputs = node->outputs();
    auto result_num = op->getNumResults();
    for (int i{0}; i < result_num; i++) {
      value_map[outputs[i]->uniqueName()] = op->getResult(i);
    }
    add_op_name_attr(op, node->name());
    block->push_back(op);
  }
  INFO(IMPORTER) << "----- building return op -----";
  llvm::SmallVector<mlir::Value> results;
  for (auto out : outputs) {
    results.push_back(value_map[out->uniqueName()]);
  }
  auto return_op = build_op<mlir::func::ReturnOp>(builder, results);
  DEBUG_BUILDED_OP(IMPORTER, return_op);
  block->push_back(return_op);
  return func;
}

mlir::ModuleOp OnnxImporter::mlir_gen(
    mlir::OpBuilder *builder, const ONNX_NAMESPACE::ModelProto &model) const {
  INFO(IMPORTER) << "----- building module op -----";
  auto loc = builder->getUnknownLoc();
  auto module = builder->create<mlir::ModuleOp>(loc);
  add_gloabal_layout_attr(module, LAYOUT::NCHW);
  DEBUG_BUILDED_OP(IMPORTER, module);
  auto graph = ONNX_NAMESPACE::ImportModelProto(model);
  auto func = mlir_gen(builder, *graph);
  module.push_back(func);
  return module;
}

mlir::ModuleOp OnnxImporter::export_mlir_module() const {
  auto builder = builder_->builder();
  auto module = mlir_gen(&builder, model_);
  INFO(IMPORTER) << "build module successfully!";
  return module;
}
};  // namespace llc::front
