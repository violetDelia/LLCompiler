// RUN: llc-opt --split-input-file --operation-legalization %s | FileCheck %s

// CHECK-LABEL: weight_refine
func.func @weight_refine() ->(tensor<f32>)  attributes {entrance}{
  // CHECK: llh.weight
  // CHECK-SAME: -> tensor<1xf32>
  %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<f32>
  return %0: tensor<f32>
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

// -----
// CHECK-LABEL: broadcast_to
func.func @broadcast_to(%arg0: tensor<?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}) -> tensor<?x?x?xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c0, value = 1 : i64}> : () -> i64
    %2 = "llh.dim"(%arg0, %1) <{symbol = @s0}> : (tensor<?x?xf32>, i64) -> i64
    %3 = "llh.dim"(%arg0, %0) <{symbol = @s1}> : (tensor<?x?xf32>, i64) -> i64
    // CHECK: llh.reshape
    %4 = "llh.reshape"(%arg0, %0, %2, %3) : (tensor<?x?xf32>, i64, i64, i64) -> tensor<1x?x?xf32>
    // CHECK: llh.reshape
    // CHECK-SAME: tensor<1x?x?xf32>
    // CHECK: llh.broadcast_to
    // CHECK-SAME:  <{cast_dims = array<i64: 0, 1, 2>}> : (tensor<1x?x?xf32>, i64, i64, i64) -> tensor<1x?x?xf32>
    %5 = "llh.broadcast_to"(%arg0, %0, %2, %3) <{cast_dims = array<i64: 1, 2>}> : (tensor<?x?xf32>, i64, i64, i64) -> tensor<1x?x?xf32>
    %6 = "llh.add"(%5, %4) : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    // CHECK: llh.reshape
    %7 = "llh.reshape"(%arg0, %2, %0, %3) : (tensor<?x?xf32>, i64, i64, i64) -> tensor<?x1x?xf32>
    // CHECK: llh.reshape
    // CHECK-SAME: tensor<1x?x?xf32>
    // CHECK: llh.broadcast_to
    // CHECK-SAME:  <{cast_dims = array<i64: 0, 1, 2>}> : (tensor<1x?x?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    %8 = "llh.broadcast_to"(%arg0, %2, %2, %3) <{cast_dims = array<i64: 1, 2>}> : (tensor<?x?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    %9 = "llh.broadcast_to"(%7, %2, %2, %3) <{cast_dims = array<i64: 0, 1, 2>}> : (tensor<?x1x?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    %10 = "llh.add"(%8, %9) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %11 = "llh.broadcast_to"(%6, %2, %2, %3) <{cast_dims = array<i64: 0, 1, 2>}> : (tensor<1x?x?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    %12 = "llh.add"(%11, %10) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %13 = "llh.extract"(%12, %0) : (tensor<?x?x?xf32>, i64) -> tensor<?x?xf32>
    %14 = "llh.extract"(%13, %0) : (tensor<?x?xf32>, i64) -> tensor<?xf32>
    // CHECK: llh.reshape
    // CHECK-SAME: tensor<1x1x?xf32>
    // CHECK: llh.broadcast_to
    // CHECK-SAME: <{cast_dims = array<i64: 0, 1, 2>}> : (tensor<1x1x?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    %15 = "llh.broadcast_to"(%14, %2, %2, %3) <{cast_dims = array<i64: 2>}> : (tensor<?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    return %15 : tensor<?x?x?xf32>
}

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --operation-legalization /home/lfr/LLCompiler/test/Dialect/LLH/operation-legalization.mlir