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
#ifndef LLC_IREX_ATTRIBUTES
#define LLC_IREX_ATTRIBUTES

include "Dialect/IRExtension/IR/Enums.td"
include "Dialect/IRExtension/IR/Constraints.td"


class IRExtension_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<IRExtension_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "mlir_ex." # attrMnemonic;
  let genStorageClass = 1;
  
}

// def IRExtension_Encoding : IRExtension_Attr<"Encoding", "encoding", []> {
//   let description = [{
//   "tensor encoding"
//   }];
//   let parameters = (ins "mlir::ArrayAttr":$shapes);
//   let hasCustomAssemblyFormat = 1;
//   // let assemblyFormat = [{
//   //   `<`
//   //     `layout` `=` $layout
//   //   `>`
//   // }];
// }

#endif // LLC_IREX_ATTRIBUTES