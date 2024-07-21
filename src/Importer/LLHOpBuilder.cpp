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
#include <cstdio>
#include <vector>

#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h"
#include "llcompiler/Dialect/Utility/Type.h"
#include "llcompiler/Importer/LLHOpBuilder.h"
#include "llcompiler/Importer/OnnxImporter.h"
#include "llcompiler/Importer/Utility.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/VectorBuilder.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "onnx/common/ir.h"
#include "onnx/common/tensor.h"

#define LLCOMPILER_ONNX_NODE_MLIR_GEN_HEAD \
  auto inputs = node->inputs();            \
  auto outputs = node->outputs();

#define LLCOMPILER_ONNX_NODE_MLIR_GEN_VALUE(name, value_set, index, value_map) \
  mlir::Value #name;                                                           \
  auto #name##_name = #value_set[#index]->uniqueName();                        \
  if (value_map.find(#name##_name) != value_map.end()) {                       \
    #name = value_map[#name##_name];                                           \
  } else {                                                                     \
    print_info << "no";                                                        \
  }

namespace llc::importer {
LLHOpBuilder::LLHOpBuilder(mlir::MLIRContext* context) : OpBuilder(context) {
  context->getOrLoadDialect<mlir::llc::llh::LLHDialect>();
}

LLCOMPILER_OPBULDER_MLIRGEN_IMPL(LLHOpBuilder, ONNX_NAMESPACE::Graph) {
  std::map<std::string, mlir::Value> value_map;
  // make funcop
  auto inputs = item.inputs();
  auto func_inputs = gen_types<OnnxImporter>(&builder_, item.inputs());
  auto func_outputs = gen_types<OnnxImporter>(&builder_, item.outputs());
  auto func_type = builder_.getFunctionType(func_inputs, func_outputs);
  auto func = builder_.create<mlir::func::FuncOp>(builder_.getUnknownLoc(),
                                                  item.name(), func_type);
  auto block = func.addEntryBlock();
  auto inputs_size = inputs.size();
  for (int i = 0; i < inputs_size; ++i) {
    std::string name = inputs[i]->uniqueName();
    value_map[name] = block->getArgument(i);
  }
  // make weight op
  std::set<std::string> weight_set;
  for (auto weight_name : item.initializer_names()) {
    weight_set.insert(weight_name);
  }
  std::map<std::string, mlir::ShapedType> weight_shape_map;
  for (auto node : item.nodes()) {
    for (auto input : node->inputs()) {
      auto name = input->uniqueName();
      if (weight_set.find(name) != weight_set.end()) {
        weight_shape_map[name] = OnnxImporter::gen_type(&builder_, input);
      }
    }
  }
  for (auto weight : item.initializers()) {
    auto weight_op = gen_mlir_(weight, weight_shape_map);
    block->push_back(weight_op);
    value_map[weight.name()] = weight_op.getResult();
  }

  auto gen_mlir = [this, &item, &block,
                   &value_map](const ONNX_NAMESPACE::Node* node) {
    if (strcmp(node->kind().toString(), "Conv") == 0) {
    } else {
      block->push_back(builder_.create<mlir::llc::llh::UndefinedOp>(
          builder_.getUnknownLoc(), node->kind().toString()));
    }
  };

  for (auto node : item.nodes()) {
    gen_mlir(node);
    //
  }
  // for (auto tensor : item.initializer_names()) {
  //   print_info << tensor;
  // }

  // func.addEntryBlock();
  // for (int i = 0; i < item.inputs().size(); ++i) {
  //   std::string name = item.inputs()[i]->uniqueName();
  //   // m_value2value[g->inputs()[i]] = entryBlock->getArgument(i);
  //   func.setArgAttr(i, "name", builder_.getStringAttr(name));
  // }
  // std::unordered_map<std::string, ONNX_NAMESPACE::Value*> init_map;
  // for (auto node : item.nodes()) {
  //   auto inputs = node->inputs();
  //   auto outputs = node->outputs();
  //   for (auto input : inputs) init_map.emplace(input->uniqueName(), input);
  //   for (auto output : outputs) init_map.emplace(output->uniqueName(),
  //   output);
  // }

  // std::unordered_map<ONNX_NAMESPACE::Value*, ONNX_NAMESPACE::Tensor>
  // tensor_map; save initializers as ParamStorage and load by ParamProvider
  // int size = item.initializers().size(); for (int i = 0; i < size; ++i) {
  //   std::string initializer_name = item.initializer_names()[i];
  //   ONNX_NAMESPACE::Tensor initializer = item.initializers()[i];
  //   ONNX_NAMESPACE::Value* init_value = init_map.at(initializer_name);
  //   tensor_map.emplace(init_value, initializer);
  //   auto storage = create_param_storage(initializer, init_value);
  //   mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
  //       m_builder.getUnknownLoc(), storage);
  //   m_value2value[init_value] = value;
  // }

  // func.setFunctionType(func_type);
  module->push_back(func);
}
#define CREATE_WEIGHT_OP_FROME_ONNX(Onnx_Type, Type, weight, shape, builder) \
  if (weight.elem_type() == Onnx_Type) {                                     \
    auto element_size = helper::get_element_size_form(shape);                \
    auto data_begin = weight.data<Type>();                                   \
    return builder.create<mlir::llc::llh::WeightOp>(                         \
        builder.getUnknownLoc(),                                             \
        mlir::DenseElementsAttr::get(                                        \
            shape,                                                           \
            llvm::ArrayRef<Type>(data_begin, data_begin + element_size)));   \
  }

mlir::llc::llh::WeightOp LLHOpBuilder::gen_mlir_(
    const ONNX_NAMESPACE::Tensor& weight,
    std::map<std::string, mlir::ShapedType>& weight_shape_map) {
  auto name = weight.name();
  auto shape = weight_shape_map[name];
  CHECK(IMPORTER, shape.hasStaticShape()) << "it not static shape for weight";
  auto elem_type = shape.getElementType();
  auto data = weight.data<int>();
  if (elem_type != OnnxImporter::gen_type(&builder_, weight.elem_type())) {
    WARN(IMPORTER) << "element_type of initializer is conflict!";
  }
  CREATE_WEIGHT_OP_FROME_ONNX(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float,
                              weight, shape, builder_)
  CREATE_WEIGHT_OP_FROME_ONNX(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
                              double, weight, shape, builder_)
  CREATE_WEIGHT_OP_FROME_ONNX(ONNX_NAMESPACE::TensorProto_DataType_INT32,
                              int32_t, weight, shape, builder_)
  CREATE_WEIGHT_OP_FROME_ONNX(ONNX_NAMESPACE::TensorProto_DataType_INT64,
                              int64_t, weight, shape, builder_)
  CREATE_WEIGHT_OP_FROME_ONNX(ONNX_NAMESPACE::TensorProto_DataType_UINT64,
                              uint64_t, weight, shape, builder_)
  UNIMPLEMENTED(IMPORTER) << "unimplemented weight data type: "
                          << weight.elem_type();
}

#undef CREATE_WEIGHT_OP_FROME_ONNX
};  // namespace llc::importer
