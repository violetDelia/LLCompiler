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
#include <string>

#include "Frontend/Core/Base.h"
#include "Frontend/Core/Option.h"
#include "llvm/Support/CommandLine.h"

namespace llc::option {

front::FrontEndOption get_front_end_option() {
  return {.input_file = option::inputFile,
          .output_file = option::outputFile,
          .onnx_convert = option::onnxConvert,
          .onnx_convert_version = option::onnxConvertVersion,
          .frontend_type = option::frontendType};
}

llvm::cl::OptionCategory importingOptions("importer options",
                                          "config for importing models");

llvm::cl::opt<front::FRONTEND_TYPE> frontendType(
    "import-type", llvm::cl::desc("the type of modle how to import"),
    llvm::cl::values(clEnumValN(front::FRONTEND_TYPE::ONNX_FILE, "onnx_file",
                                "onnx model")),
    llvm::cl::init(front::FRONTEND_TYPE::ONNX_FILE),
    llvm::cl::cat(importingOptions));

llvm::cl::opt<std::string> inputFile(
    "input-file", llvm::cl::desc("the path of the file to import"),
    llvm::cl::value_desc("model file"), llvm::cl::init(""),
    llvm::cl::cat(importingOptions));

llvm::cl::opt<std::string> outputFile(
    "output-file", llvm::cl::desc("the path of the file to import"),
    llvm::cl::value_desc("model file"), llvm::cl::init(""),
    llvm::cl::cat(importingOptions));

llvm::cl::opt<bool> onnxConvert(
    "onnx-convert",
    llvm::cl::desc("onnx version convert will make some error in "
                   "ONNX_NAMESPACE::ConvertVersion,it "
                   "best to input onnx file verison of 22"),
    llvm::cl::value_desc("version"), llvm::cl::init(false),
    llvm::cl::cat(importingOptions));

llvm::cl::opt<uint64_t> onnxConvertVersion(
    "onnx-version-to",
    llvm::cl::desc("the onnx model will convert to onnx-version before "
                   "lowering to dialect,default convert to 16"),
    llvm::cl::value_desc("version"), llvm::cl::init(22),
    llvm::cl::cat(importingOptions));

}  // namespace llc::option
