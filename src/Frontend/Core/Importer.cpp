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
#include <utility>

#include "llcompiler/Frontend/Core/Base.h"
#include "llcompiler/Frontend/Core/Importer.h"
#include "mlir/IR/MLIRContext.h"


namespace llc::front {

Importer::Importer(mlir::MLIRContext *context, const ImporterOption &options)
    : builder_(context), option_(std::move(options)) {}

Importer::~Importer() {}

};  // namespace llc::front