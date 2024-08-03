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

#include "llcompiler/Support/Logger.h"
#include "llcompiler/Support/Option.h"
#include "llvm/Support/CommandLine.h"
namespace llc::option {
llvm::cl::OptionCategory commonOptions{"global options", ""};

llvm::cl::opt<std::string> logRoot{
    "log-root", llvm::cl::desc("the root to save log files"),
    llvm::cl::value_desc("root_path"), llvm::cl::init(""),
    llvm::cl::cat(commonOptions)};

llvm::cl::opt<logger::LOG_LEVEL> logLevel{
    "log-lever", llvm::cl::desc("log level"),
    llvm::cl::values(
        clEnumValN(logger::LOG_LEVEL::DEBUG_,
                   logger::log_level_to_str(logger::LOG_LEVEL::DEBUG_), ""),
        clEnumValN(logger::LOG_LEVEL::INFO_,
                   logger::log_level_to_str(logger::LOG_LEVEL::INFO_), ""),
        clEnumValN(logger::LOG_LEVEL::WARN_,
                   logger::log_level_to_str(logger::LOG_LEVEL::WARN_), ""),
        clEnumValN(logger::LOG_LEVEL::ERROR_,
                   logger::log_level_to_str(logger::LOG_LEVEL::ERROR_), ""),
        clEnumValN(logger::LOG_LEVEL::FATAL_,
                   logger::log_level_to_str(logger::LOG_LEVEL::FATAL_), "")),
    llvm::cl::init(logger::LOG_LEVEL::DEBUG_), llvm::cl::cat(commonOptions)};

logger::LoggerOption get_logger_option() {
  return logger::LoggerOption{.path = logRoot, .level = logLevel};
}

}  // namespace llc::option
