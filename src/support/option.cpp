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

#include <string>

#include "llcompiler/support/core.h"
#include "llcompiler/support/logger.h"
#include "llcompiler/support/option.h"
#include "llvm/Support/CommandLine.h"

namespace llc::option {
llvm::cl::OptionCategory commonOptions("global options", "");

llvm::cl::opt<std::string> logRoot("log-root",
                                   llvm::cl::desc("the root to save log files"),
                                   llvm::cl::value_desc("root_path"),
                                   llvm::cl::init("C:/LLCompiler/log"),
                                   llvm::cl::cat(commonOptions));

llvm::cl::opt<logger::LOG_LEVER> logLevel(
    "log-lever", llvm::cl::desc("log level"),
    llvm::cl::values(
        clEnumValN(logger::LOG_LEVER::DEBUG,
                   logger::log_lever_to_str(logger::LOG_LEVER::DEBUG), ""),
        clEnumValN(logger::LOG_LEVER::INFO,
                   logger::log_lever_to_str(logger::LOG_LEVER::INFO), ""),
        clEnumValN(logger::LOG_LEVER::WARN,
                   logger::log_lever_to_str(logger::LOG_LEVER::WARN), ""),
        clEnumValN(logger::LOG_LEVER::ERROR,
                   logger::log_lever_to_str(logger::LOG_LEVER::ERROR), ""),
        clEnumValN(logger::LOG_LEVER::FATAL,
                   logger::log_lever_to_str(logger::LOG_LEVER::FATAL), "")),
    llvm::cl::init(logger::LOG_LEVER::DEBUG), llvm::cl::cat(commonOptions));

llvm::cl::OptionCategory importingOptions("importer options",
                                          "config for importing models");

llvm::cl::opt<importer::IMPORTER_TYPE> importingType(
    "import-type", llvm::cl::desc("the type of modle how to import"),
    llvm::cl::values(clEnumValN(importer::IMPORTER_TYPE::ONNX_FILE, "onnx_file",
                                "onnx model")),
    llvm::cl::init(importer::IMPORTER_TYPE::ONNX_FILE),
    llvm::cl::cat(importingOptions));

llvm::cl::opt<std::string> importingPath(
    "import-file", llvm::cl::desc("the path of the file to import"),
    llvm::cl::value_desc("model file"),
    llvm::cl::init("C:/LLCompiler/models/resnet18-v1-7.onnx"),
    llvm::cl::cat(importingOptions));

}  // namespace llc::option
