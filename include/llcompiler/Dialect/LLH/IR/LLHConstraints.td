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
#ifndef LLH_CONSTRAINTS
#define LLH_CONSTRAINTS
include "llcompiler/Dialect/LLH/IR/LLHTypes.td"

def LLH_AnyType   : Type<CPred<"true">, "any type">;

def LLH_Bool      : I<1>;

def LLH_Int8      : I<8>;
def LLH_Int16     : I<16>;
def LLH_Int32     : I<32>;
def LLH_Int64     : I<64>;

def LLH_UInt8     : SI<8>;
def LLH_UInt16    : SI<16>;
def LLH_UInt32    : SI<32>;
def LLH_UInt64    : SI<64>;

def LLH_SInt      : Type<CPred<"$_self.isSignlessInteger()">, "signless integer","::mlir::IntegerType">;
def LLH_UInt      : Type<CPred<"$_self.isUnsignedInteger()">, "unsigned integer">;
def LLH_Int       : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self)">, "integer","::mlir::IntegerType">;

def LLH_F16       : F<16>;
def LLH_F32       : F<32>;
def LLH_F64       : F<64>;
def LLH_BF16      : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
                    BuildableType<"$_builder.getBF16Type()">;

def LLH_Float     : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self)">, "floating-point","::mlir::FloatType">;

def LLH_BoolTensor    : TensorOf<[LLH_Bool]>;

def LLH_Int8Tensor    : TensorOf<[LLH_Int8]>;
def LLH_Int16Tensor   : TensorOf<[LLH_Int16]>;
def LLH_Int32Tensor   : TensorOf<[LLH_Int32]>;
def LLH_Int64Tensor   : TensorOf<[LLH_Int64]>;


def LLH_UInt8Tensor   : TensorOf<[LLH_UInt8]>;
def LLH_UInt16Tensor  : TensorOf<[LLH_UInt16]>;
def LLH_UInt32Tensor  : TensorOf<[LLH_UInt32]>;
def LLH_UInt64Tensor  : TensorOf<[LLH_UInt64]>;

def LLH_SIntTensor    : TensorOf<[LLH_SInt]>;
def LLH_UIntTensor    : TensorOf<[LLH_UInt]>;
def LLH_IntTensor     : TensorOf<[LLH_Int]>;

def LLH_F16Tensor     : TensorOf<[LLH_F16]>;
def LLH_F32Tensor     : TensorOf<[LLH_F32]>;
def LLH_F64Tensor     : TensorOf<[LLH_F64]>;
def LLH_BF16Tensor    : TensorOf<[LLH_BF16]>;

def LLH_FloatTensor   : TensorOf<[LLH_Float]>;

def LLH_Tensor        : AnyTypeOf<[LLH_FloatTensor,LLH_IntTensor]>;

def LLH_ScalarTensor  : Type<And<[IsTensorTypePred,
                                  CPred<"::llvm::cast<::mlir::TensorType>($_self).getRank() == 0">]>>;

def LLH_Rank1BoolTensor : TensorOf<[LLH_Bool], [CPred<"::llvm::cast<::mlir::TensorType>($_self).getRank() == 1">]>;


def LLH_String : StringBasedAttr<CPred<"::llvm::isa<::mlir::StringAttr>($_self)">,
                              "string attribute">;

class DenseElementsAttrBase<Pred condition, string summary> :
    Attr<condition, summary> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];
  let convertFromStorage = "$_self";
}

def DenseElementsAttr : DenseElementsAttrBase<CPred<"::llvm::isa<::mlir::DenseElementsAttr>($_self)">,
                                    "constant vector/tensor attribute">;

#endif // LLH_CONSTRAINTS