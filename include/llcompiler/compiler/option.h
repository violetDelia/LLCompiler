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
#ifndef INCLUDE_LLCOMPILER_COMPILER_OPTION_H_
#define INCLUDE_LLCOMPILER_COMPILER_OPTION_H_

#include "llcompiler/core.h"
#include "llcompiler/utils/logger.h"
#include "llvm/Support/CommandLine.h"

namespace llc::option {
ALIAS_CLASS_1(Option, llvm::cl::opt)
ALIAS_CLASS(Category, llvm::cl::OptionCategory)
using logger::LOG_LEVER;

}  // namespace llc::option

extern llc::option::Category LLC_CommonOption_Cat;

extern llc::option::Option<llc::String> LLC_logRoot;
extern llc::option::Option<llc::option::LOG_LEVER> LLC_logLevel;
#endif  // INCLUDE_LLCOMPILER_COMPILER_OPTION_H_
