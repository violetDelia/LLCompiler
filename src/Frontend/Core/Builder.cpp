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

#include "llcompiler/Frontend/Core/Builder.h"

#include "llcompiler/Dialect/IRExtension/IR/Dialect.h"
#include "llcompiler/Dialect/LLH/IR/LLHDialect.h"
#include "llcompiler/Support/Logger.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"

namespace llc::front {
#define LOAD_DIALECT(Dialect)           \
  context->getOrLoadDialect<Dialect>(); \
  INFO(IMPORTER) << "load dialect: " << Dialect::getDialectNamespace().str();

Builder::Builder(mlir::MLIRContext* context) : builder_(context) {
  LOAD_DIALECT(mlir::BuiltinDialect);
  LOAD_DIALECT(mlir::func::FuncDialect);
  LOAD_DIALECT(mlir::tosa::TosaDialect);
  LOAD_DIALECT(mlir::llh::LLHDialect);
  LOAD_DIALECT(mlir::ex::IRExtensionDialect);
}

#undef LOAD_DIALECT
Builder::~Builder() {}

mlir::OpBuilder& Builder::builder() { return builder_; }

};  // namespace llc::front
