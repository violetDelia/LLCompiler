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
#ifndef LLC_EUNMS
#define LLC_EUNMS

include "mlir/IR/EnumAttr.td"


def LLC_Mode_Trainning         : I32EnumAttrCase<"training", 0>;
def LLC_Mode_Inference         : I32EnumAttrCase<"inference", 1>;
def LLC_Mode : I32EnumAttr<"ModeKind", "Mode",
    [LLC_Mode_Trainning, LLC_Mode_Inference]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llc";
}

def LLC_Target_X86                :I32EnumAttrCase<"x86", 0>;
def LLC_Target_X86_64             :I32EnumAttrCase<"x86_64", 1>;
def LLC_Target: I32EnumAttr<"Target", "Target",
    [LLC_Target_X86,LLC_Target_X86_64]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llc";
}

def LLC_Log_Level_Debug           :I32EnumAttrCase<"debug", 0>;
def LLC_Log_Level_Info            :I32EnumAttrCase<"info", 1>;
def LLC_Log_Level_Warning         :I32EnumAttrCase<"warn", 2>;
def LLC_Log_Level_Error           :I32EnumAttrCase<"error", 3>;
def LLC_Log_Level_Fatal           :I32EnumAttrCase<"fatal", 4>;
def LLC_Log_Level: I32EnumAttr<"LogLevel", "LogLevel",
    [LLC_Log_Level_Debug, LLC_Log_Level_Info, LLC_Log_Level_Warning, LLC_Log_Level_Error, LLC_Log_Level_Fatal]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llc";
}

def LLC_Global_Layout_NCHW    :I32EnumAttrCase<"NCHW", 0>;
def LLC_Global_Layout_NHWC    :I32EnumAttrCase<"NHWC", 1>;
def LLC_Global_Layout: I32EnumAttr<"GlobalLayout", "GlobalLayout",
    [LLC_Global_Layout_NCHW, LLC_Global_Layout_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::llc";
}
#endif //