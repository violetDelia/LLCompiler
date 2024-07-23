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
#ifndef LLH_EUNMS
#define LLH_EUNMS
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

def LLH_AUTO_PAD_NOTSET         : I32EnumAttrCase<"DEFAULT", 0>;
def LLH_AUTO_PAD_SAME_UPPER     : I32EnumAttrCase<"SAME_UPPER", 1>;
def LLH_AUTO_PAD_SAME_LOWER     : I32EnumAttrCase<"SAME_LOWER", 2>;
def LLH_AUTO_PAD_VALID          : I32EnumAttrCase<"VALID", 3>;
def LLH_AutoPad : I32EnumAttr<"AutoPad",
    "auto pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.",
    [LLH_AUTO_PAD_NOTSET, LLH_AUTO_PAD_SAME_UPPER, LLH_AUTO_PAD_SAME_LOWER, LLH_AUTO_PAD_VALID]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::llc::llh";
}
def LLH_AutoPadAttr : EnumAttr<LLH_Dialect, LLH_AutoPad, "AutoPad">;

def LLH_CEIL_MODE_CEIL        : I32EnumAttrCase<"CEIL", 0>;
def LLH_CEIL_MODE_FLOOR       : I32EnumAttrCase<"FLOOR", 1>;
def LLH_CeilMode : I32EnumAttr<"CeilMode",
    "Whether to use ceil or floor (default) to compute the output shape.",
    [LLH_CEIL_MODE_CEIL, LLH_CEIL_MODE_FLOOR]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::llc::llh";
}
def LLH_CeilModeAttr : EnumAttr<LLH_Dialect, LLH_CeilMode, "CeilMode">;

def LLH_Shift_Right         : I32EnumAttrCase<"Right", 0>;
def LLH_Shift_Left          : I32EnumAttrCase<"Left", 1>;
def LLH_ShiftDirection : I32EnumAttr<"ShiftDirection",
    "Direction of moving bits. It can be either “RIGHT” (for right shift) or “LEFT” (for left shift)..",
    [LLH_Shift_Right, LLH_Shift_Left]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::llc::llh";
}
def LLH_ShiftDirectionAttr : EnumAttr<LLH_Dialect, LLH_ShiftDirection, "ShiftDirection">;
#endif //