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
#ifndef LLH_LLHTOTOSA
#define LLH_LLHTOTOSA
include "mlir/IR/PatternBase.td"
include "mlir/IR/Constraints.td"
include "llcompiler/Dialect/LLH/IR/LLHOps.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"


def Gen_Dense_Form_Attr: NativeCodeCall<"llc::genDenseElementsFromArrayAttr($0)">;

def Gen_Zore_ElementAttr: NativeCodeCall<"genZoreElementAttr($0)">;

def Clone_Tensor_With_Layout_Any: NativeCodeCall<"cloneTensorWithLayoutAny($0)">;

def Lowing_TransposeOp :
    Pat<    (LLH_TransposeOp:$res $X, $perms),
            (Tosa_TransposeOp $X, 
                (Tosa_ConstOp (Gen_Dense_Form_Attr $perms))
            )>;

def Lowing_WeightOp :
    Pat<    (LLH_WeightOp:$res $value),
            (Tosa_ConstOp $value)
    >;

def Lowing_ConstantOp :
    Pat<    (LLH_ConstantOp:$res $value),
            (Tosa_ConstOp $value)
    >;

def Lowing_ReluOp :
    Pat<    (LLH_ReluOp:$res $x),
            (Tosa_MaximumOp $x,
                (Tosa_ConstOp (Gen_Zore_ElementAttr $res),
                    (returnType  (Clone_Tensor_With_Layout_Any $res))
                )
            )
    >;
#endif // LLH_LLHTOTOSA