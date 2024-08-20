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
#ifndef LLH_TRANSFORMLAYOUTTONHWC
#define LLH_TRANSFORMLAYOUTTONHWC
include "mlir/IR/PatternBase.td"
include "mlir/IR/Constraints.td"
include "llcompiler/Dialect/LLH/IR/LLHOps.td"

def Gen_Transpose_Perms_To_NHWC: NativeCodeCall<"genTransposePermsToNHWC($0)">;

def Gen_Transpose_Perms_From_NHWC: NativeCodeCall<"genTransposePermsFromNHWC($0)">;

def Gen_NHWC_Return_Tensor: NativeCodeCall<"genNHWCReturnTensor($0)">;

def Layout_Is_Not_NHWC: Constraint<CPred<"layoutIsNotNHWC($0)">>;

def Any_Layout_Is_Not_NHWC_3:
    Constraint< Or<[    CPred<"layoutIsNotNHWC($0)">,
                        CPred<"layoutIsNotNHWC($1)">,
                        CPred<"layoutIsNotNHWC($2)">]>>;


def Is_Same_Layout_3: 
    Constraint< And<[   CPred<"layoutIsSame($0,$1)">,
                        CPred<"layoutIsSame($1,$2)">]>>;


def Transform_ConvOp_To_NHWC : 
    Pat<  (LLH_ConvOp:$res $X, $W, $B, $dilation, $kernel_shape, $pad, $stride, $group),
          (LLH_TransposeOp
            (LLH_ConvOp 
                (LLH_TransposeOp $X, 
                    (Gen_Transpose_Perms_To_NHWC $X), 
                    (returnType (Gen_NHWC_Return_Tensor $X))),
                (LLH_TransposeOp $W, 
                    (Gen_Transpose_Perms_To_NHWC $W),
                    (returnType (Gen_NHWC_Return_Tensor $W))), 
                $B, $dilation, $kernel_shape, $pad, $stride, $group,
                (returnType (Gen_NHWC_Return_Tensor $res))
            ),
            (Gen_Transpose_Perms_From_NHWC $res)),
        [(Any_Layout_Is_Not_NHWC_3 $X, $W, $res), (Is_Same_Layout_3 $X, $W, $res)]>;

#endif // LLH_TRANSFORMLAYOUTTONHWC