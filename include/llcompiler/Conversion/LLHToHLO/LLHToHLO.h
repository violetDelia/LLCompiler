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

#ifndef INCLUDE_LLCOMPILER_CONVERSION_LLHTOHLO_LLHTOHLO_H_
#define INCLUDE_LLCOMPILER_CONVERSION_LLHTOHLO_LLHTOHLO_H_
#include <memory>
namespace mlir {
class MLIRContext;
class TypeConverter;
class Pass;
class RewritePatternSet;
class ConversionTarget;
#define GEN_PASS_DECL_CONVERTLLHTOHLOPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir
#endif  // INCLUDE_LLCOMPILER_CONVERSION_LLHTOHLO_LLHTOHLO_H_
