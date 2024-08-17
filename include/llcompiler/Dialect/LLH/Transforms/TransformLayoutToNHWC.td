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
include "llcompiler/Dialect/LLH/IR/LLHOps.td"

// def Transform_ConvOp_To_NHWC : 
//     Pattern<    (LLH_ConvOp $X, $W, $B, $dilation, $kernel_shape, $pad, $stride, $group),
//                 (LLH_ConvOp (LLH_TransposeOp($X )), $W, $B, $dilation, $kernel_shape, $pad, $stride, $group)>;

#endif // LLH_TRANSFORMLAYOUTTONHWC