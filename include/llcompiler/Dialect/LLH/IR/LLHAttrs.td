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
#ifndef LLH_ATTRIBUTES
#define LLH_ATTRIBUTES

include "llcompiler/Dialect/LLH/IR/LLHEnums.td"
include "llcompiler/Dialect/LLH/IR/LLHConstraints.td"

class LLH_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<LLH_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "llh." # attrMnemonic;
  let genStorageClass = 1;
  
}

def LLH_Encoding : LLH_Attr<"Encoding", "encoding", []> {
  let description = [{
  "tensor encoding"
  }];
  let parameters = (ins ArrayRefParameter<"mlir::FlatSymbolRefAttr", "">:$shape_symbols);
  let builders = [
    AttrBuilder<(ins "mlir::ArrayRef<mlir::StringRef>":$shape_symbols), [{
      mlir::SmallVector<mlir::FlatSymbolRefAttr> symbols;
      for (auto sym : shape_symbols) {
        symbols.push_back(
            FlatSymbolRefAttr::get($_ctxt, StringAttr::get($_ctxt, sym)));
      }
      return $_get($_ctxt, symbols);
    }]>,
  ];
  //let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    `<`
      `shapes` `=` $shape_symbols
    `>`
  }];
}

#endif // LLH_ATTRIBUTES