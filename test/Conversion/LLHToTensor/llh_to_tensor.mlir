// RUN: llc-opt --split-input-file --convert-llh-to-tensor %s| FileCheck %s

// CHECK-LABEL: constant
func.func @constant() ->() attributes {entrance}{
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}