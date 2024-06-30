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

namespace llc::option {

ALIAS_FUNCTION(InitAttr, llvm::cl::init)
using DescAttr = llvm::cl::desc;
using ValueDescAttr = llvm::cl::value_desc;
using CatAtrr = llvm::cl::cat;
ALIAS_FUNCTION(ValuesAttr, llvm::cl::values)

}  // namespace llc::option

llc::option::Category LLC_CommonOption_Cat("global options", "");

llc::option::Option<llc::String> LLC_logRoot(
    "log-root", llc::option::DescAttr("the root to save log files"),
    llc::option::ValueDescAttr("root_path"), llc::option::InitAttr(""),
    llc::option::CatAtrr(LLC_CommonOption_Cat));

llc::option::Option<llc::option::LOG_LEVER> LLC_logLevel(
    "log-lever", llc::option::DescAttr("log level"),
    llc::option::ValuesAttr(
        clEnumValN(llc::option::LOG_LEVER::DEBUG, "debug", ""),
        clEnumValN(llc::option::LOG_LEVER::INFO, "info", ""),
        clEnumValN(llc::option::LOG_LEVER::WARN, "warning", ""),
        clEnumValN(llc::option::LOG_LEVER::ERROR, "error", ""),
        clEnumValN(llc::option::LOG_LEVER::FATAL, "fatal", "")),
    llc::option::InitAttr(llc::option::LOG_LEVER::DEBUG),
    llc::option::CatAtrr(LLC_CommonOption_Cat));
