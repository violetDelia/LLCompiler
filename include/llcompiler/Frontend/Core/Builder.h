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
 * @file Opbuilder.h
 * @brief interface class OpBuilder that build Ops form inputs.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#ifndef INCLUDE_LLCOMPILER_FRONTEND_CORE_BUILDER_H_
#define INCLUDE_LLCOMPILER_FRONTEND_CORE_BUILDER_H_

namespace llc::front {

class Builder {
 public:
  explicit Builder(mlir::MLIRContext* context);
  virtual ~Builder();
  mlir::OpBuilder& builder();

 protected:
  mlir::OpBuilder builder_;
};

}  // namespace llc::front
#endif  // INCLUDE_LLCOMPILER_FRONTEND_CORE_BUILDER_H_
