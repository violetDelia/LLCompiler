// SPDX-License-Identifier: Apache-2.0

//===--- AttrBase.td -- Base Attribute Definitions -----------*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors
//
// =============================================================================
//
// This file contains the base set of constructs for defining Attribute classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLC_IREXTENSION_ENCODEING
#define LLC_IREXTENSION_ENCODEING
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"

def Builtin_LAYOUT_NCHW         : I32EnumAttrCase<"NCHW", 0>;
def Builtin_LAYOUT_NHWC        : I32EnumAttrCase<"NHWC", 1>;
def Builtin_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [Builtin_LAYOUT_NCHW, Builtin_LAYOUT_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::ex";
}
def Builtin_LayoutAttr : EnumAttr<Builtin_Dialect, Builtin_Layout, "Layout">;


def Builtin_Encoding : AttrDef<Builtin_Dialect, "Encoding", [], "::mlir::Attribute"> {
  let mnemonic = ?;
  let attrName = "builtin_ex.encoding";
  let summary = "tensor encoding";
  let description = [{
  }];
  let parameters = (ins Builtin_LayoutAttr:$layout);
}

#endif // LLC_IREXTENSION_ENCODEING
