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
#include <any>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "llcompiler/Compiler/Init.h"

// namespace llc::importer
int main(int argc, char **argv) {
  llc::init_compiler(argc, argv);

  //  ONNX_NAMESPACE::ModelProto model;
  //  ONNX_NAMESPACE::LoadProtoFromPath(llc::option::importingPath.getValue(),
  //                                    model);
  //  ONNX_NAMESPACE::shape_inference::InferShapes(model);
  //  std::unique_ptr<ONNX_NAMESPACE::Graph> g(
  //      ONNX_NAMESPACE::ImportModelProto(model));
  //  std::cout << g->docString() << std::endl;
  //  auto nodes = g.nodes();

  // auto in =  g->inputs();
  // auto out = g->outputs();
  // std::cout<<g->name()<<std::endl;
  return 0;
}
