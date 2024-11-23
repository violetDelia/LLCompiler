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
include "Dialect/LLH/IR/LLHDialect.td"


//===----------------------------------------------------------------------===//
// LLH type definitions
//===----------------------------------------------------------------------===//

class LLH_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLH_Dialect, name, traits,"::mlir::Type"> {
  let mnemonic = typeMnemonic;
  let typeName = dialect.name # "." # typeMnemonic;
}


class LLH_Tensor<string name, string typeMnemonic,list<Trait> traits = []>
    : LLH_Type<name, typeMnemonic, traits> {
  let summary = "dynamic Tensor type";
  let description = [{
    
  }];
  let parameters = (ins
    ArrayRefParameter<"DynamicDim">:$dims,
    "::mlir::Type":$type
  );
  let genStorageClass = 0;
  let skipDefaultBuilders = 0;
  let genVerifyDecl = 1;
  let assemblyFormat = "`<`custom<DynamicDim>($dims)``$type`>`";
  let builders = [
    TypeBuilder<(ins
      "::mlir::ArrayRef<int64_t>":$dims,
      "::mlir::Type":$type
    ), [{
      llvm::SmallVector<DynamicDim, 4> dynamic_dims;
      for (auto i : dims){
        if(i < 0){
          dynamic_dims.push_back(DynamicDim(-1,true));
        }
        else{
          dynamic_dims.push_back(DynamicDim(i));
        }
      }
      return $_get(context, dynamic_dims, type);
    }]>
  ];
  string extraBaseClassDeclaration = [{
  }];
}


def LLH_TensorType : LLH_Tensor<"Tensor","tensor">;

def LLH_IntType : LLH_Type<"Int", "int"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    int type;
    !llh.int<width>; 
    !llh.int<"u"width>;
  }];
  let parameters = (ins "unsigned":$width, "SIGNED_TAG":$signed_tag
  );
  let genStorageClass = 1;
  let skipDefaultBuilders = 0;
  let genVerifyDecl = 1;
  let assemblyFormat = "`<`custom<SIGNED_TAG>($signed_tag)``$width`>`";
  let builders = [
    TypeBuilder<(ins "unsigned":$width,
                     CArg<"bool", "true">:$is_signed),[{
      if(is_signed) return Base::get($_ctxt,width,SIGNED);
      return Base::get($_ctxt,width,UNSIGNED);
    }]>
  ];
  let extraClassDeclaration = [{
    /// Return true if this is a signless integer type.
    bool isSignless() const { return getSignedTag() == SIGNLESS; }
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getSignedTag() == SIGNED; }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return getSignedTag() == UNSIGNED; }

    /// Integer representation maximal bitwidth.
    /// Note: This is aligned with the maximum width of llvm::IntegerType.
    static constexpr unsigned Max_Width = (1 << 24) - 1;
  }];
}



#endif // LLH_TYPES