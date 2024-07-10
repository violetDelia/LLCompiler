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
#include "iostream"
#include "llcompiler/Compiler/Init.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Importer/OnnxImporter.h"
#include "llcompiler/Importer/Utility.h"
#include "llcompiler/Support/Core.h"

int main(int argc, char **argv) {
  llc::init_compiler(argc, argv);
  mlir::MLIRContext context;
  context.getOrLoadDialect<llc::llh::LLHDialect>();
  auto import_option = llc::importer::get_importer_option();
  auto module = llc::importer::gen_mlir_from_to(&context, import_option);
  return 0;
}
