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
#ifndef LLH_DIALECT
#define LLH_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLH dialect definition.
//===----------------------------------------------------------------------===//

def LLH_Dialect : Dialect{
    let name = "llh";
    let summary = "Top-level dialect for connecting LLC and other graph frameworks";
    let description = [{
     Top-level dialect for connecting LLC and other graph frameworks.
     }];
    let cppNamespace = "::mlir::llh";
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];
    let dependentDialects = [];
    
}


//===----------------------------------------------------------------------===//
// Base LLH operation trait definition.
//===----------------------------------------------------------------------===//

class LLH_OpTrait<string name> : NativeOpTrait<""> {
  let trait = name;
  let cppNamespace = "::llc::llh::OpTrait";
}

#endif // LLH_DIALECT
