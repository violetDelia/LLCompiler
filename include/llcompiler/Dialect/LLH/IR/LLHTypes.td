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
include "llcompiler/Dialect/LLH/IR/LLHDialect.td"

//===----------------------------------------------------------------------===//
// LLH type definitions
//===----------------------------------------------------------------------===//

class LLH_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LLH_Dialect, name, traits,"::mlir::Type"> {
  let mnemonic = typeMnemonic;
  let typeName = "llc." # typeMnemonic;
}

def LLH_IntType : LLH_Type<"Int", "int"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    Syntax:

    ```
    // Sized integers like i1, i4, i8, i16, i32.
    signed-integer-type ::= `si` [1-9][0-9]*
    unsigned-integer-type ::= `ui` [1-9][0-9]*
    signless-integer-type ::= `i` [1-9][0-9]*
    integer-type ::= signed-integer-type |
                     unsigned-integer-type |
                     signless-integer-type
    ```

    Integer types have a designated bit width and may optionally have signedness
    semantics.

    **Rationale:** low precision integers (like `i2`, `i4` etc) are useful for
    low-precision inference chips, and arbitrary precision integers are useful
    for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller
    than a 16 bit one).
  }];
  let parameters = (ins "unsigned":$width, "SIGNED_TAG":$signed_tag
  );
  let builders = [
    TypeBuilder<(ins "unsigned":$width,CArg<"SIGNED_TAG", "SIGNLESS">:$signed_tag
    )>
  ];

  // IntegerType uses a special storage class that compacts parameters to save
  // memory.
  let genStorageClass = 1;
  let skipDefaultBuilders = 0;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    
    /// Return true if this is a signless integer type.
    //bool isSignless() const { return getSignedness() == Signless; }
    /// Return true if this is a signed integer type.
    //bool isSigned() const { return getSignedness() == Type::Signed; }
    /// Return true if this is an unsigned integer type.
    //bool isUnsigned() const { return getSignedness() == Type::Unsigned; }

    /// Get or create a new IntegerType with the same signedness as `this` and a
    /// bitwidth scaled by `scale`.
    /// Return null if the scaled element type cannot be represented.
    //IntegerType scaleElementBitwidth(unsigned scale);

    /// Integer representation maximal bitwidth.
    /// Note: This is aligned with the maximum width of llvm::IntegerType.
    static constexpr unsigned Max_Width = (1 << 24) - 1;
  }];
  //let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "`<`$width``custom<SIGNED_TAG>($signed_tag)`>`";
  //let assemblyFormat = "`<` $width static_cast<int>($signed_tag)`>`";
}

//===----------------------------------------------------------------------===//
//  definitions type constraints
//===----------------------------------------------------------------------===//
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

def LLH_Tensor        : TensorOf<[LLH_AnyType]>;

def LLH_Index   : Type<CPred<"::llvm::isa<::mlir::IndexType>($_self)">, "index", "::mlir::IndexType">,
                  BuildableType<"$_builder.getIndexType()">;




#endif // LLH_TYPES