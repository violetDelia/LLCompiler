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
#ifndef LLH_TOSAEX_DIALECT
#define LLH_TOSAEX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TosaEx dialect definition.
//===----------------------------------------------------------------------===//

def TosaEx_Dialect : Dialect{
    let name = "tosa_ex";
    let summary = "extension of tosa dialect";
    let description = [{
     extension of tosa dialect. if front can't convert to tasa op, use this.
     }];
    let cppNamespace = "::mlir::tosa_ex";
    let dependentDialects = ["tensor::TensorDialect", "quant::QuantizationDialect", "tosa::TosaDialect"];
    let useDefaultAttributePrinterParser = 0;
    let hasConstantMaterializer = 0;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    
}

//===----------------------------------------------------------------------===//
// Base TosaEx operation definition.
//===----------------------------------------------------------------------===//

class TosaEx_Op<string mnemonic, list<Trait> traits = []> :
        Op<TosaEx_Dialect, mnemonic, traits>;
#endif // LLH_TOSAEX_DIALECT
