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

#include "include/llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Compiler/Utility.h"
#include "llcompiler/Frontend/Core/Option.h"
#include "llcompiler/Frontend/Core/Utility.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/IR/MLIRContext.h"
#define LLCOMPILER_HAS_LOG
// namespace llc::importer
int main(int argc, char **argv) {
  llc::front::FrontEndOption options{
      .input_file = "C:/LLCompiler/example/models/mnist-12.onnx",
      .onnx_convert_version = 16,
      .frontend_type = llc::front::FRONTEND_TYPE::ONNX_FILE};
  mlir::MLIRContext context;
  auto module = llc::compiler::gen_mlir_from(&context, options);
  module->dump();
  return 0;
}
