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
//

#ifndef LLH_CANONICALIZE
#define LLH_CANONICALIZE

include "Dialect/LLH/IR/LLHOps.td"
include "Dialect/Utility/TensorPred.td"
include "mlir/IR/PatternBase.td"


//===----------------------------------------------------------------------===//
// Common utility functions.
//===----------------------------------------------------------------------===//
def getRankOf :
	NativeCodeCall<"mlir::cast<ShapedType>($0.getType()).getRank()">;

def AttributeIsNotNull :
  Constraint<CPred<"($_self)">, "Attribute is not null">;

def IsDenseElementsAttr :
  Constraint<And<[CPred<" ($_self) ">,
                  CPred<"mlir::isa<DenseElementsAttr>(($_self))">
                 ]>, "Attribute is not a DenseElementsAttr">;

def GetNullAttr : NativeCodeCall<"Attribute()">;

def GetNullFloatAttr : NativeCodeCall<"FloatAttr()">;

def GetNullIntegerAttr : NativeCodeCall<"IntegerAttr()">;

def GetNullStringAttr : NativeCodeCall<"StringAttr()">;

def GetNullArrayAttr :  NativeCodeCall<"ArrayAttr()">;

class HasRankGT<int rank> :
  Constraint<CPred<"mlir::isa<ShapedType>($0.getType()) && "
                   "mlir::cast<ShapedType>($0.getType()).hasRank() && "
                   "mlir::cast<ShapedType>($0.getType()).getRank() > " # rank>>;
class HasRankOf<int rank> :
  Constraint<CPred<"mlir::isa<ShapedType>($0.getType()) && "
                   "mlir::cast<ShapedType>($0.getType()).hasRank() && "
                   "mlir::cast<ShapedType>($0.getType()).getRank() == " # rank>>;


def HaveSameShapedType: Constraint<
    CPred<"(isa<ShapedType>($0.getType()) &&"
          "dyn_cast<ShapedType>($0.getType()) == "
          "dyn_cast<ShapedType>($1.getType()))">,
    "has same shaped type">;


def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def HasNoneType : Constraint<CPred<"mlir::isa<NoneType>($0.getType())">>;

def NotNoneType : Constraint<CPred<"!mlir::isa<NoneType>(($0.getType()))">>;

def HaveSameElementType : Constraint<
    CPred<"(mlir::dyn_cast<ShapedType>($0.getType()).getElementType() == "
          "mlir::dyn_cast<ShapedType>($1.getType()).getElementType())">,
    "have same element types">;

def HaveSameElementTypeBitWidth: Constraint<
    CPred<"(mlir::dyn_cast<ShapedType>($0.getType()).getElementTypeBitWidth() == "
          "mlir::dyn_cast<ShapedType>($1.getType()).getElementTypeBitWidth())">,
    "has same element type bitwidth">;

def IsStaticShapeTensor:
  Constraint<
    CPred<
      "mlir::cast<::mlir::ShapedType>($_self.getType()).hasStaticShape()">,
    "hasStaticShape">;

def Equal: Constraint<CPred<"$0 == $1">, "are equal">;

class EqualString<string s> : Constraint<CPred<"$0 == \"" # s # "\"">>;


//===----------------------------------------------------------------------===//
// Canonicalization for AbsOp
//===----------------------------------------------------------------------===//
def FoldTwoAbsOpPattern : Pat<(LLH_AbsOp (LLH_AbsOp:$res $input)),
                                 (LLH_AbsOp $input),
                                 [(HasOneUse $res)]>;

#endif // LLH_CANONICALIZE
