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
include "llcompiler/Dialect/Utility/TensorPred.td"
include "mlir/Dialect/Tosa/IR/TosaOps.td"


def Gen_Dense_Form_Attr: NativeCodeCall<"llc::genDenseElementsFromArrayAttr($0)">;

def Gen_Zore_ElementAttr: NativeCodeCall<"genZoreElementAttr($0)">;

def Clone_Tensor_With_Layout_Any: NativeCodeCall<"cloneTensorWithLayoutAny($0)">;

def Get_Shape: NativeCodeCall<"GetShape($0)">;

def Gen_Unsqueeze_Dim0_Shape: NativeCodeCall<"UnsqueezeShape($0, 0)">;

def Gen_Squeeze_Dim0_Shape: NativeCodeCall<"SqueezeShape($0, 0)">;

def Gen_Unsqueeze_Dim0_Tensor: NativeCodeCall<"UnsqueezeTensor($0, 0)">;

def Gen_Squeeze_Dim0_Tensor: NativeCodeCall<"SqueezeTensor($0, 0)">;

// def Lowing_TransposeOp :
//     Pat<    (LLH_TransposeOp:$res $X, $perms),
//             (Tosa_TransposeOp $X, 
//                 (Tosa_ConstOp (Gen_Dense_Form_Attr $perms))
//             )>;

// def Lowing_WeightOp :
//     Pat<    (LLH_WeightOp:$res $value),
//             (Tosa_ConstOp $value)
//     >;

// def Lowing_ConstantOp :
//     Pat<    (LLH_ConstantOp:$res $value),
//             (Tosa_ConstOp $value)
//     >;

// def Lowing_ReluOp :
//     Pat<    (LLH_ReluOp:$res $x),
//             (Tosa_MaximumOp $x,
//                 (Tosa_ConstOp (Gen_Zore_ElementAttr $res),
//                     (returnType  (Clone_Tensor_With_Layout_Any $res))
//                 )
//             )
//     >;

// def Lowing_MatmalOp_3D :
//     Pat<    (LLH_MatMulOp:$res $lhs, $rhs),
//             (Tosa_MatMulOp $lhs ,$rhs),
//         [(Is_3D_Tensor $lhs), (Is_3D_Tensor $rhs), (Is_3D_Tensor $res)]
//     >;

// def Lowing_MatmalOp_2D:
//     Pat<    (LLH_MatMulOp:$res $lhs, $rhs),
//             (Tosa_ReshapeOp 
//                 (Tosa_MatMulOp 
//                     (Tosa_ReshapeOp $lhs, (Gen_Unsqueeze_Dim0_Shape $lhs),
//                         (returnType (Gen_Unsqueeze_Dim0_Tensor $lhs))
//                     ),
//                     (Tosa_ReshapeOp $rhs, (Gen_Unsqueeze_Dim0_Shape $lhs),
//                         (returnType (Gen_Unsqueeze_Dim0_Tensor $lhs))
//                     ),
                    
//                     (returnType (Gen_Unsqueeze_Dim0_Tensor $res))
//                 ),
//                 (Get_Shape $res)   
//             ),
//         [(Is_2D_Tensor $lhs), (Is_2D_Tensor $rhs), (Is_2D_Tensor $res)]
//     >;
#endif // LLH_LLHTOTOSA