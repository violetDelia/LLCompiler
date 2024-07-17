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

#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHOps.h"
#include "llcompiler/Dialect/LLH/IR/LLHTypes.h"
#include "llcompiler/Importer/LLHOpBuilder.h"
#include "llcompiler/Importer/OnnxImporter.h"
#include "llcompiler/Importer/Utility.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/VectorBuilder.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "onnx/common/ir.h"

namespace llc::importer {
LLHOpBuilder::LLHOpBuilder(mlir::MLIRContext* context) : OpBuilder(context) {
  context->getOrLoadDialect<llc::llh::LLHDialect>();
}

LLCOMPILER_OPBULDER_MLIRGEN_IMPL(LLHOpBuilder, onnx::Graph) {
  std::map<std::string, mlir::Value> value_map;
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

  auto gen_mlir = [this, block, value_map](const onnx::Node* node) {
    if (strcmp(node->kind().toString(), "Conv") == 0) {
      // auto inputs = node->inputs();
      // auto outputs = node->outputs();
      // auto input_1_name = inputs[0]->uniqueName();
      // auto input_2_name = inputs[1]->uniqueName();
      // auto output_1_name = outputs[0]->uniqueName();
      // print_info << input_1_name;
      // mlir::Value input_1 = value_map.at(input_1_name);
      // mlir::Value input_2 = value_map.at(input_2_name);
      // auto output_1 = OnnxImporter::gen_type(&builder_, outputs[0]);
      // block->push_back(builder_.create<llc::llh::AddOp>(
      //     builder_.getUnknownLoc(), output_1, input_1, input_2));
      // if (value_map.count(input_1_name)) {
      //   input_1 = value_map[input_1_name];
      // } else {
      //   input_2
      // }

    } else {
      block->push_back(builder_.create<llc::llh::UndefinedOp>(
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
  // std::unordered_map<std::string, onnx::Value*> init_map;
  // for (auto node : item.nodes()) {
  //   auto inputs = node->inputs();
  //   auto outputs = node->outputs();
  //   for (auto input : inputs) init_map.emplace(input->uniqueName(), input);
  //   for (auto output : outputs) init_map.emplace(output->uniqueName(),
  //   output);
  // }

  // std::unordered_map<onnx::Value*, onnx::Tensor> tensor_map;
  // save initializers as ParamStorage and load by ParamProvider
  // int size = item.initializers().size();
  // for (int i = 0; i < size; ++i) {
  //   std::string initializer_name = item.initializer_names()[i];
  //   onnx::Tensor initializer = item.initializers()[i];
  //   onnx::Value* init_value = init_map.at(initializer_name);
  //   tensor_map.emplace(init_value, initializer);
  //   auto storage = create_param_storage(initializer, init_value);
  //   mlir::Value value = m_builder.create<mlir::MGB::ParamProvider>(
  //       m_builder.getUnknownLoc(), storage);
  //   m_value2value[init_value] = value;
  // }

  // func.setFunctionType(func_type);
  module->push_back(func);
}

};  // namespace llc::importer
