// RUN: llc-opt --split-input-file --operation-legalization %s | FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --operation-legalization /home/lfr/LLCompiler/test/Dialect/LLH/operation-legalization.mlir
module attributes {builtin.gloabal_layout = "NCHW"} {
  // CHECK-LABEL: add_layout_attr
  func.func @add_layout_attr(%arg0: tensor<?x3x?x?xf32>) ->()  attributes {entrance}{
  %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<64x3x7x7xf32>
  // CHECK: llh.conv
  // CHECK-SAME: layout = #llh.Layout<NCHW>
  %124 = "llh.conv"(%arg0, %0) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<?x64x?x?xf32>
  // CHECK: llh.max_pool
  // CHECK-SAME: layout = #llh.Layout<NCHW>
  %129 = "llh.max_pool"(%124) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<?x64x?x?xf32>
  return 
  }
}

// -----
module attributes {builtin.gloabal_layout = "NCHW"} {
// CHECK-LABEL: weight_refine
func.func @weight_refine() ->()  attributes {entrance}{
  // CHECK: llh.weight
  // CHECK-SAME: -> tensor<1xf32>
  %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<f32>
  return 
}
}