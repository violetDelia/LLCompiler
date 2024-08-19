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
#ifndef LLC_IREX_EUNMS
#define LLC_IREX_EUNMS
include "mlir/IR/EnumAttr.td"
include "llcompiler/Dialect/IRExtension/IR/Dialect.td"

def IRExtension_LAYOUT_NCHW         : I32EnumAttrCase<"NCHW", 0>;
def IRExtension_LAYOUT_NHWC        : I32EnumAttrCase<"NHWC", 1>;
def IRExtension_LAYOUT_Any        : I32EnumAttrCase<"Any", 2>;
def IRExtension_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [IRExtension_LAYOUT_NCHW, IRExtension_LAYOUT_NHWC, IRExtension_LAYOUT_Any]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::ex";
}

def IRExtension_LayoutAttr : EnumAttr<IRExtension_Dialect, IRExtension_Layout, "Layout">{
  let assemblyFormat = "`<` $value `>`";
}

#endif //LLC_IREX_EUNMS