// RUN: llc-opt --split-input-file --reshape-before-braodcast %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --reshape-before-braodcast /home/lfr/LLCompiler/test/Dialect/LLH/reshape_before_braodcast.mlir
module attributes {builtin.gloabal_layout = "NCHW"} {
  "llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
  // CHECK-LABEL: simplyBinary
  func.func @simplyBinary(%arg0: tensor<?x?x?x?xf32>) -> () attributes {entrance} {
    %1 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    // CHECK: llh.reshape
    // CHECK-SAME: -> tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>
    %29 = "llh.add"(%arg0, %1) : (tensor<?x?x?x?xf32>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<?x?x?x?xf32>
    return 
  }
}
