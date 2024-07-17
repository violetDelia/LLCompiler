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
#ifndef LLH_TYPE_CONSTRAINTS
#define LLH_TYPE_CONSTRAINTS
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

def LLH_String : StringBasedAttr<CPred<"::llvm::isa<::mlir::StringAttr>($_self)">,
                              "string attribute">;
// def LLH_Index   : Type<CPred<"::llvm::isa<::mlir::IndexType>($_self)">, "index", "::mlir::IndexType">,
//                   BuildableType<"$_builder.getIndexType()">;


// //===----------------------------------------------------------------------===//
// //  definitions type constraints
// //===----------------------------------------------------------------------===//

// def LLH_IsIntTypePred : CPred<"::llvm::isa<::llc::llh::IntType>($_self)">;

// //===----------------------------------------------------------------------===//
// //  definitions type constraints
// //===----------------------------------------------------------------------===//
// class SignedInt<int width>
//     : Type< And<[  LLH_IsIntTypePred,
//                   CPred<"llvm::cast<::llc::llh::IntType>($_self).getWidth() =="#width>,
//                   CPred<"llvm::cast<::llc::llh::IntType>($_self).getSignedTag() == SIGNED"> ]>,
//             width # "-bit signed integer", "::llc::llh::IntType">,
//       BuildableType<"$_builder.getIntType(" # width # ", /*is_signed=*/true)">{
//   int bitwidth = width;
// }

// class UnSignedInt<int width>
//     : Type< And<[  LLH_IsIntTypePred,
//                   CPred<"llvm::cast<::llc::llh::IntType>($_self).getWidth() =="#width>,
//                   CPred<"llvm::cast<::llc::llh::IntType>($_self).getSignedTag() == UNSIGNED"> ]>,
//             width # "-bit signed integer", "::llc::llh::IntType">,
//       BuildableType<"$_builder.getIntType(" # width # ", /*is_signed=*/false)">{
//   int bitwidth = width;
// }

// //===----------------------------------------------------------------------===//
// // type constraints
// //===----------------------------------------------------------------------===//


// def LLH_AnyType   : Type<CPred<"true">, "any type">;

// def LLH_Bool      : Type<And<[  LLH_IsIntTypePred,
//                                 CPred<"llvm::cast<::llc::llh::IntType>($_self).getWidth() =="#width>]>>;

// def LLH_Int8      : SignedInt<8>;
// def LLH_Int16     : SignedInt<16>;
// def LLH_Int32     : SignedInt<32>;
// def LLH_Int64     : SignedInt<64>;

// def LLH_UInt8     : UnSignedInt<8>;
// def LLH_UInt16    : UnSignedInt<16>;
// def LLH_UInt32    : UnSignedInt<32>;
// def LLH_UInt64    : UnSignedInt<64>;

// def LLH_SInt      : Type<CPred<"$_self.isSignlessInteger()">, "signless integer","::mlir::IntegerType">;
// def LLH_UInt      : Type<CPred<"$_self.isUnsignedInteger()">, "unsigned integer">;
// def LLH_Int       : Type<LLH_IsIntTypePred>;

// def LLH_F16       : F<16>;
// def LLH_F32       : F<32>;
// def LLH_F64       : F<64>;
// def LLH_BF16      : Type<CPred<"$_self.isBF16()">, "bfloat16 type">,
//                     BuildableType<"$_builder.getBF16Type()">;

// def LLH_Float     : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self)">, "floating-point","::mlir::FloatType">;

// def LLH_BoolTensor    : TensorOf<[LLH_Bool]>;

// def LLH_Int8Tensor    : TensorOf<[LLH_Int8]>;
// def LLH_Int16Tensor   : TensorOf<[LLH_Int16]>;
// def LLH_Int32Tensor   : TensorOf<[LLH_Int32]>;
// def LLH_Int64Tensor   : TensorOf<[LLH_Int64]>;


// def LLH_UInt8Tensor   : TensorOf<[LLH_UInt8]>;
// def LLH_UInt16Tensor  : TensorOf<[LLH_UInt16]>;
// def LLH_UInt32Tensor  : TensorOf<[LLH_UInt32]>;
// def LLH_UInt64Tensor  : TensorOf<[LLH_UInt64]>;

// def LLH_SIntTensor    : TensorOf<[LLH_SInt]>;
// def LLH_UIntTensor    : TensorOf<[LLH_UInt]>;
// def LLH_IntTensor     : TensorOf<[LLH_Int]>;

// def LLH_F16Tensor     : TensorOf<[LLH_F16]>;
// def LLH_F32Tensor     : TensorOf<[LLH_F32]>;
// def LLH_F64Tensor     : TensorOf<[LLH_F64]>;
// def LLH_BF16Tensor    : TensorOf<[LLH_BF16]>;

// def LLH_FloatTensor   : TensorOf<[LLH_Float]>;

// def LLH_Tensor        : TensorOf<[LLH_AnyType]>;

// def LLH_Index   : Type<CPred<"::llvm::isa<::mlir::IndexType>($_self)">, "index", "::mlir::IndexType">,
//                   BuildableType<"$_builder.getIndexType()">;

#endif // LLH_TYPE_CONSTRAINTS