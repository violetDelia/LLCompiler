// RUN: llc-opt --split-input-file --canonicalize %s| FileCheck %s

// CHECK-LABEL: dim_to_const
func.func @dim_to_const(%101: tensor<?x512x1x1xf32>) ->(i64, i64, i64, i64) attributes {entrance} {
  // CHECK-COUNT-3: llh.constant
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  // CHECK-COUNT-1: llh.dim
  %102 = "llh.dim"(%101, %2) : (tensor<?x512x1x1xf32>, i64) -> i64
  %103 = "llh.dim"(%101, %3) : (tensor<?x512x1x1xf32>, i64) -> i64
  %104 = "llh.dim"(%101, %1) : (tensor<?x512x1x1xf32>, i64) -> i64
  %105 = "llh.dim"(%101, %0) : (tensor<?x512x1x1xf32>, i64) -> i64
  return %102,%103,%104,%105: i64, i64, i64, i64
}

// -----
// CHECK-LABEL: fold_two_abs
func.func @fold_two_abs(%arg0: tensor<?x?x?x?xf32>) ->  tensor<?x?x?x?xf32> attributes {entrance} {
  // CHECK: llh.abs
  // CHECK: return
  %4 = "llh.abs"(%arg0) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %5 = "llh.abs"(%4) : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %5 : tensor<?x?x?x?xf32>
}

// -----
// CHECK-LABEL: decompose_extract
func.func @decompose_extract(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s3"}) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = -1 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = -5 : i64}> : () -> i64
  // CHECK: llh.dim
  // CHECK: llh.add
  // CHECK: llh.extract
  %6 = "llh.extract"(%arg0, %0) : (tensor<?x?x?x?xf32>, i64) -> tensor<?x?x?xf32>
  // CHECK: llh.dim
  // CHECK: llh.add
  // CHECK: llh.extract
  %7 = "llh.extract"(%arg0, %1) : (tensor<?x?x?x?xf32>, i64) -> tensor<?x?x?xf32>
  return 
}

// -----
// CHECK-LABEL: fold_reshape
func.func @fold_reshape() -> (tensor<1x10xf32>) attributes {entrance} {
    %1 = "llh.constant"() <{value = 10 : i64}> : () -> i64
    %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %5 = "llh.constant"() <{value = dense<[-0.00187645969, -0.0539493635, -0.0634634792, -0.0399834365, -0.0204222016, -0.030699648, 0.044618234, 0.0177833326, 0.0345337801, 0.0643900782]> : tensor<10xf32>}> : () -> tensor<10xf32>
    // CHECK-NOT: llh.reshape
    %11 = "llh.reshape"(%5, %3, %1) : (tensor<10xf32>, i64, i64) -> tensor<1x10xf32>
    return %11 : tensor<1x10xf32>
}

// -----
// CHECK-LABEL: fold_broadcast
func.func @fold_broadcast(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 2 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
    %3 = "llh.dim"(%arg0, %2) : (tensor<?x?x?xf32>, i64) -> i64
    %4 = "llh.dim"(%arg0, %1) : (tensor<?x?x?xf32>, i64) -> i64
    %5 = "llh.dim"(%arg0, %0) : (tensor<?x?x?xf32>, i64) -> i64
    %6 = "llh.transpose"(%arg0) <{perms = array<i64: 0, 2, 1>}> : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %7 = "llh.broadcast_to"(%arg0, %3, %4, %5) <{cast_dims = array<i64: 0, 1, 2>, expand_dims = array<i64>, noexpand_dims = array<i64: 0, 1, 2>}> : (tensor<?x?x?xf32>, i64, i64, i64) -> tensor<?x?x?xf32>
    // CHECK: return %arg0
    return %7 : tensor<?x?x?xf32>
  }

// -----
// CHECK-LABEL: fold_convert_to
func.func @fold_convert_to(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {entrance} {
    %0 = "llh.convert_to"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    // CHECK: return %arg0
    return %0 : tensor<?x?x?xf32>
  }

// -----
// CHECK-LABEL: fold_reshape
func.func @fold_reshape(%arg0: tensor<200x3x224x224xf32>) -> tensor<200x3x224x224xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 200 : i64}> : () -> i64
    %1 = "llh.constant"() <{value = 3 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = 224 : i64}> : () -> i64
    %33 = "llh.reshape"(%arg0, %0, %1, %2, %2) : (tensor<200x3x224x224xf32>, i64, i64, i64, i64) -> tensor<200x3x224x224xf32>
    // CHECK: return %arg0
    return %33 : tensor<200x3x224x224xf32>
  }

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file -canonicalize /home/lfr/LLCompiler/test/Dialect/LLH/canonicalize.mlir 



