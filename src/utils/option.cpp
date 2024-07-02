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
                                   llvm::cl::init(""),
                                   llvm::cl::cat(commonOptions));

llvm::cl::opt<LOG_LEVER> logLevel(
    "log-lever", llvm::cl::desc("log level"),
    llvm::cl::values(clEnumValN(LOG_LEVER::DEBUG, "debug", ""),
                     clEnumValN(LOG_LEVER::INFO, "info", ""),
                     clEnumValN(LOG_LEVER::WARN, "warning", ""),
                     clEnumValN(LOG_LEVER::ERROR, "error", ""),
                     clEnumValN(LOG_LEVER::FATAL, "fatal", "")),
    llvm::cl::init(LOG_LEVER::DEBUG), llvm::cl::cat(commonOptions));
}  // namespace llc::option
