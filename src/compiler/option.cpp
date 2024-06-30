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

#include "llcompiler/compiler/option.h"

namespace llc::option {}  // namespace llc::option

llvm::cl::OptionCategory LLC_CommonOption_Cat("global options", "");

llvm::cl::opt<std::string> LLC_logRoot(
    "log-root", llvm::cl::desc("the root to save log files"),
    llvm::cl::value_desc("root_path"), llvm::cl::init(""),
    llvm::cl::cat(LLC_CommonOption_Cat));

llvm::cl::opt<llc::logger::LOG_LEVER> LLC_logLevel(
    "log-lever", llvm::cl::desc("log level"),
    llvm::cl::values(clEnumValN(llc::logger::LOG_LEVER::DEBUG, "debug", ""),
                     clEnumValN(llc::logger::LOG_LEVER::INFO, "info", ""),
                     clEnumValN(llc::logger::LOG_LEVER::WARN, "warning", ""),
                     clEnumValN(llc::logger::LOG_LEVER::ERROR, "error", ""),
                     clEnumValN(llc::logger::LOG_LEVER::FATAL, "fatal", "")),
    llvm::cl::init(llc::logger::LOG_LEVER::DEBUG),
    llvm::cl::cat(LLC_CommonOption_Cat));
