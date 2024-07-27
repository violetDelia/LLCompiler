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
 * @file Importer.h
 * @brief Importer can convert input to mlir::ModuleOp,this is a interface
 * class.
 * @author 时光丶人爱 (1733535832@qq.com)
 * @version 1.0
 * @date 2024-07-01
 *
 * @copyright Copyright (c) 2024 时光丶人爱
 *
 */

#include "llcompiler/Frontend/Core/Base.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"


#ifndef INCLUDE_LLCOMPILER_FRONTEND_CORE_IMPORTER_H_
#define INCLUDE_LLCOMPILER_FRONTEND_CORE_IMPORTER_H_

namespace llc::front {

class Importer {
 public:
  Importer(mlir::MLIRContext *context, const ImporterOption &option);
  virtual ~Importer();
  virtual mlir::ModuleOp export_mlir_module() const = 0;

 protected:
  template <class Importer, class Container>
  auto gen_types(mlir::Builder *builder, const Container &container);

 protected:
  mlir::Builder builder_;
  const ImporterOption option_;
};

template <class Importer, class Container>
auto gen_types(mlir::Builder *builder, const Container &container) {
  return llvm::to_vector<1>(
      llvm::map_range(container, [builder](auto value) -> mlir::Type {
        return gen_type(builder, value);
      }));
}
}  // namespace llc::front

#endif  // INCLUDE_LLCOMPILER_FRONTEND_CORE_IMPORTER_H_
