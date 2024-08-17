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
#ifndef LLC_IREX_DIALECT
#define LLC_IREX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IR extension dialect definition.
//===----------------------------------------------------------------------===//

def IRExtension_Dialect : Dialect{
    let name = "mlir_ex";
    let summary = "mlir extension";
    let description = [{
     mlir extension.
     }];
    let cppNamespace = "::mlir::ex";
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 0;
    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];
    
}


#endif // LLC_IREX_
