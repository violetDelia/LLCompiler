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

def Gen_Transpose_Perms_NCHW_To_NHWC: NativeCodeCall<"genTransposePermsToNHWC($0,llc::LAYOUT::NCHW)">;

def Gen_Transpose_Perms_NCHW_From_NHWC: NativeCodeCall<"genTransposePermsFromNHWC($0,llc::LAYOUT::NCHW)">;

def Gen_Return_Tensor_For_NCHW_To_NHWC: NativeCodeCall<"genReturnTensorFrom($0,llc::LAYOUT::NCHW)">;

def Has_LayoutAttr_NCHW: Constraint<CPred<"HaslLayoutAttr($0,llc::LAYOUT::NCHW)">>;


def Transform_ConvOp_NCHW_To_NHWC : 
    Pat<  (LLH_ConvOp:$res $X, $W, $B, $dilation, $kernel_shape, $pad, $stride, $group),
          (LLH_TransposeOp
            (LLH_ConvOp 
                (LLH_TransposeOp $X, 
                    (Gen_Transpose_Perms_NCHW_To_NHWC $X), 
                    (returnType (Gen_Return_Tensor_For_NCHW_To_NHWC $X))),
                (LLH_TransposeOp $W, 
                    (Gen_Transpose_Perms_NCHW_To_NHWC $W),
                    (returnType (Gen_Return_Tensor_For_NCHW_To_NHWC $W))), 
                $B, $dilation, $kernel_shape, $pad, $stride, $group,
                (returnType (Gen_Return_Tensor_For_NCHW_To_NHWC $res))
            ),
            (Gen_Transpose_Perms_NCHW_From_NHWC $res)),
        [(Has_LayoutAttr_NCHW $res)]>;

#endif // LLH_TRANSFORMLAYOUTTONHWC