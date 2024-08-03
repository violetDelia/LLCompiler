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

#include "llcompiler/Importer/OpBuilder.h"
#include "llcompiler/Support/Core.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
namespace llc::importer {
OpBuilder::OpBuilder(mlir::MLIRContext* context) : builder_(context) {
  context->getOrLoadDialect<mlir::func::FuncDialect>();
}

OpBuilder::~OpBuilder() {}

mlir::OpBuilder& OpBuilder::builder() { return builder_; }

OpBuilderTrace::OpBuilderTrace(OpBuilder* builder) : builder_(builder) {}

OpBuilderTrace::~OpBuilderTrace() {}

mlir::OpBuilder& OpBuilderTrace::builder() const { return builder_->builder(); }
};  // namespace llc::importer