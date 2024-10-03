// RUN: llc-opt --split-input-file --convert-llh-to-tensor %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-tensor /home/lfr/LLCompiler/test/Conversion/LLHToTensor/llh_to_tensor.mlir

// CHECK-LABEL: dim
func.func @dim(%arg0: tensor<?x3x?x?xf32>) ->() attributes {entrance}{
  %c3_i64 = arith.constant 3 : i64
  // CHECK-COUNT: index.casts
  // CHECK-COUNT: tensor.dim
  %8 = "llh.dim"(%arg0, %c3_i64) : (tensor<?x3x?x?xf32>, i64) -> i64
  return 
}

// -----
//CHECK-LABEL: reshape
func.func @reshape(%arg0: tensor<?x?x?x?xf32>) ->() attributes {entrance}{
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %2 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %3 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %4 = "llh.constant"() <{value = 0 : i64}> : () -> i64
    %5 = "llh.dim"(%arg0, %4) : (tensor<?x?x?x?xf32>, i64) -> i64
    %6 = "llh.dim"(%arg0, %3) : (tensor<?x?x?x?xf32>, i64) -> i64
    %7 = "llh.dim"(%arg0, %2) : (tensor<?x?x?x?xf32>, i64) -> i64
    %8 = "llh.dim"(%arg0, %0) : (tensor<?x?x?x?xf32>, i64) -> i64
    // CHECK: tensor.from_elements
    // CHECK: tensor.reshape
    %9 = "llh.reshape"(%arg0, %5, %6, %7, %8) : (tensor<?x?x?x?xf32>, i64, i64, i64, i64) -> tensor<?x?x?x?xf32>
  return 
}

// -----
//CHECK-LABEL: empty
func.func @empty(%arg0: tensor<?x?x?x?xf32>) ->() attributes {entrance}{
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %2 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %3 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %4 = "llh.constant"() <{value = 0 : i64}> : () -> i64
    %5 = "llh.dim"(%arg0, %4) : (tensor<?x?x?x?xf32>, i64) -> i64
    %6 = "llh.dim"(%arg0, %3) : (tensor<?x?x?x?xf32>, i64) -> i64
    %7 = "llh.dim"(%arg0, %2) : (tensor<?x?x?x?xf32>, i64) -> i64
    %8 = "llh.dim"(%arg0, %0) : (tensor<?x?x?x?xf32>, i64) -> i64
    // CHECK: tensor.empty
    %9 = "llh.empty"(%5, %6, %7, %8) : (i64, i64, i64, i64) -> tensor<?x?x?x?xf32>
    // CHECK: tensor.empty
    %15 = "llh.empty"(%6) : (i64) -> tensor<?xf32>
  return 
}