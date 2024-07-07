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

#ifndef LLH_TYPES
#define LLH_TYPES

include "mlir/IR/DialectBase.td"
include "llcompiler/dialect/LLH/IR/LLHDialect.td"

//===----------------------------------------------------------------------===//
// Standalone type definitions
//===----------------------------------------------------------------------===//

class LLH_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLH_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// def Torch_TestType : LLH_Type<"test_type_name", "test_type_mnemonic"> {
//   let summary = "test_type";
//   let description = [{
//     test_type.
//   }];
//   let parameters = (ins StringRefParameter<"class name">:$className);
//   let hasCustomAssemblyFormat = 1;
// }

#endif // LLH_TYPES