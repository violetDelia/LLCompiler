// RUN: llc-opt --split-input-file --convert-llh-to-arith %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-arith /home/lfr/LLCompiler/test/Conversion/LLHToArith/llh_to_arith.mlir

// CHECK-LABEL: constant
func.func @constant() ->() attributes {entrance}{
  // CHECK: llh.constant
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  // CHECK: arith.constant
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}
// -----

// CHECK-LABEL: binary
func.func @binary() ->(i64) attributes {entrance}{
  // CHECK: arith.constant
  %98 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  // CHECK: arith.muli
  %106 = "llh.mul"(%98, %98): (i64, i64) -> i64
  return %106: i64
}