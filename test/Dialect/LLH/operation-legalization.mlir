// RUN: llc-opt --split-input-file --operation-legalization %s | FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --operation-legalization /home/lfr/LLCompiler/test/Dialect/LLH/operation-legalization.mlir

module attributes {builtin.gloabal_layout = "NCHW"} {
// CHECK-LABEL: weight_refine
func.func @weight_refine() ->()  attributes {entrance}{
  // CHECK: llh.weight
  // CHECK-SAME: -> tensor<1xf32>
  %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<f32>
  return 
}
}