// RUN: llc-opt --split-input-file --convert-llh-to-tosa %s| FileCheck %s

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --reshape-before-braodcast /home/lfr/LLCompiler/test/Dialect/LLH/reshape_before_braodcast.mlir
func.func @constant() ->() attributes {entrance}{
  // CHECK: tosa.const
  // CHECK-SAME: tensor<384xf32>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  // CHECK: llh.constant
  // CHECK-SAME: -> i64
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}