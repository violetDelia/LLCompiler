// RUN: llc-opt --split-input-file --operation-legalization %s | FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --operation-legalization /home/lfr/LLCompiler/test/Dialect/LLH/operation-legalization.mlir

// CHECK-LABEL: weight_refine
func.func @weight_refine() ->()  attributes {entrance}{
  // CHECK: llh.weight
  // CHECK-SAME: -> tensor<1xf32>
  %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<f32>
  return 
}
// -----
// CHECK-LABEL: extract
// CHECK-SAME: -> tensor<1xf32>
func.func @extract(%arg0: tensor<3x3xf32> {func.input_symbol_0 = "c3", func.input_symbol_1 = "c3"}) -> tensor<f32> attributes {entrance} {
    %0 = "llh.constant"() <{value = -2 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = -1 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %3 = "llh.extract"(%arg0, %2) : (tensor<3x3xf32>, i64) -> tensor<3xf32>
    %4 = "llh.extract"(%3, %1) : (tensor<3xf32>, i64) -> tensor<f32>
    %5 = "llh.mul"(%3, %4) : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
    // CHECK: llh.extract
    // CHECK: llh.extract
    // CHECK: llh.extract
    // CHECK-SAME: -> tensor<1xf32>
    %6 = "llh.extract"(%5, %0) : (tensor<3xf32>, i64) -> tensor<f32>
    return %6 : tensor<f32>
  }
