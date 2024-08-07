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
#ifndef INCLUDE_LLCOMPILER_SUPPORT_OPTION_H_
#define INCLUDE_LLCOMPILER_SUPPORT_OPTION_H_
/**
 * @file Option.h
 * @brief global options define
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */

#include <string>

#include "llcompiler/Support/Logger.h"
#include "llvm/Support/CommandLine.h"

namespace llc::option {

::llc::logger::LoggerOption get_logger_option();
extern llvm::cl::OptionCategory commonOption;
extern llvm::cl::opt<std::string> logRoot;
extern llvm::cl::opt<llc::logger::LOG_LEVEL> logLevel;
}  // namespace llc::option

#endif  // INCLUDE_LLCOMPILER_SUPPORT_OPTION_H_
