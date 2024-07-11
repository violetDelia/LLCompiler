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
/**
 * @file LLHOpbuilder.h
 * @brief implementation of Opbuilder that building LLHops.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */
#include "llcompiler/Importer/OpBuilder.h"
#include "llcompiler/Support/Core.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"


#ifndef INCLUDE_LLCOMPILER_IMPORTER_LLHOPBUILDER_H_
#define INCLUDE_LLCOMPILER_IMPORTER_LLHOPBUILDER_H_

namespace llc::importer {
class LLHOpBuilder : public OpBuilder {
 public:
  using OpBuilder::OpBuilder;
};
}  // namespace llc::importer
#endif  // INCLUDE_LLCOMPILER_IMPORTER_LLHOPBUILDER_H_
