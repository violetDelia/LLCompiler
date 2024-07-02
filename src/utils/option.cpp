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

#include "llcompiler/utils/logger.h"
#include "llcompiler/utils/option.h"
#include "llvm/Support/CommandLine.h"

namespace llc::option {
llvm::cl::OptionCategory commonOptions("global options", "");

llvm::cl::opt<std::string> logRoot("log-root",
                                   llvm::cl::desc("the root to save log files"),
                                   llvm::cl::value_desc("root_path"),
                                   llvm::cl::init("C:/LLCompiler/log"),
                                   llvm::cl::cat(commonOptions));

llvm::cl::opt<LOG_LEVER> logLevel(
    "log-lever", llvm::cl::desc("log level"),
    llvm::cl::values(clEnumValN(DEBUG, "debug", ""),
                     clEnumValN(INFO, "info", ""),
                     clEnumValN(WARN, "warning", ""),
                     clEnumValN(ERROR, "error", ""),
                     clEnumValN(FATAL, "fatal", "")),
    llvm::cl::init(DEBUG), llvm::cl::cat(commonOptions));

llvm::cl::opt<importer::IMPORTER_TYPE> importerType(
    "importer", llvm::cl::desc("the type of modle how to import"),
    llvm::cl::values(clEnumValN(importer::ONNX, "onnx", "onnx model")),
    llvm::cl::init(importer::ONNX), llvm::cl::cat(commonOptions));

}  // namespace llc::option
