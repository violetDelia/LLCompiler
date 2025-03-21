// RUN: llc-opt --split-input-file --preprocessing-llh-to-hlo --cse %s| FileCheck %s

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --preprocessing-llh-to-hlo --cse /home/lfr/LLCompiler/test/Conversion/LLHToHLO/preprocessing_llh_to_hlo.mlir

// CHECK-LABEL: relu
func.func @relu(%arg1: tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) ->(tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) attributes {entrance}{
  // CHECK-NOT: relu
  // CHECK: llh.constant
  // CHECK: llh.max
  %102 = "llh.relu"(%arg1) : (tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>) -> tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
  return %102: tensor<?x512x?x?xf32, #llh.encoding<shapes = @s0, @c512, @s42, @s43>>
}

// -----
// CHECK-LABEL: extract
func.func @extract(%arg0: tensor<?x?x?x?xf32>) -> tensor<1xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 0 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    // CHECK: llh.stride_slice
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 4, 4, 4>
    // CHECK: llh.reshape
    %2 = "llh.extract"(%arg0, %0) : (tensor<?x?x?x?xf32>, i64) -> tensor<?x?x?xf32>
    %3 = "llh.dim"(%2, %0)  : (tensor<?x?x?xf32>, i64) -> i64
    %4 = "llh.sub"(%3, %1): (i64, i64) -> i64
    // CHECK: llh.stride_slice
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 3, 3, 3>
    // CHECK: llh.reshape
    %5 = "llh.extract"(%2, %4) : (tensor<?x?x?xf32>, i64) -> tensor<?x?xf32>
    // CHECK: llh.stride_slice
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 2, 2, 2>
    // CHECK: llh.reshape
    %6 = "llh.extract"(%5, %0) : (tensor<?x?xf32>, i64) -> tensor<?xf32>
    // CHECK: llh.stride_slice
    // CHECK-NOT: llh.reshape
    %7 = "llh.extract"(%6, %0) : (tensor<?xf32>, i64) -> tensor<1xf32>
    return %7 : tensor<1xf32>
  }