// RUN: llc-opt --split-input-file --convert-llh-to-hlo %s| FileCheck %s

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-hlo /home/lfr/LLCompiler/test/Conversion/LLHToHLO/llh_to_hlo.mlir

func.func @constant() ->() attributes {entrance}{
  // CHECK: mhlo.constant
  // CHECK-SAME: tensor<384xf32>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  // CHECK: llh.constant
  // CHECK-SAME: -> i64
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}

// -----

func.func @add_and_sub(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: mhlo.subtract
  %23 = "llh.sub"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  // CHECK: mhlo.add
  %24 = "llh.add"(%23, %23) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----

func.func @mul(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: mhlo.multiply
  %23 = "llh.mul"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----

func.func @div(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  %98 = "llh.constant"() <{value = dense<1> : tensor<384xi32>}> : () -> tensor<384xi32>
  // CHECK: mhlo.divide
  %222 = "llh.div"(%98, %98) : (tensor<384xi32>, tensor<384xi32>) -> tensor<384xi32>
  return 
}

// -----

func.func @conv_nchw_fchw(%arg0: tensor<4x3x5x5xf32> , %arg1: tensor<?x3x?x?xf32>) -> (tensor<?x4x?x?xf32>) attributes {entrance} {
    // CHECK: mhlo.convolution
    // CHECK-SAME: dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
    %2 = "llh.conv"(%arg1, %arg0) <{dilation = array<i64: 2, 2>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, layout = #llh.Layout<NCHW>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<4x3x5x5xf32>) -> tensor<?x4x?x?xf32>
    return %2 : tensor<?x4x?x?xf32>
}

// -----

func.func @transpose(%arg0: tensor<?x?x?x4xf32>) -> () attributes {entrance} {
    // CHECK: mhlo.transpose
    // CHECK-SAME: (tensor<?x?x?x4xf32>) -> tensor<?x4x?x?xf32>
    %7 = "llh.transpose"(%arg0) <{perms = array<i64: 0, 3, 1, 2>}> : (tensor<?x?x?x4xf32>) -> tensor<?x4x?x?xf32>
    return 
}

// -----

func.func @braodcast_to(%arg0: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %5 = "llh.reshape"(%4, %3, %3, %3, %3) : (tensor<1xf32>, i64, i64, i64, i64) -> tensor<1x1x1x1xf32>
  %6 = "llh.dim"(%arg0, %2) : (tensor<1x?x?x?xf32>, i64) -> i64
  %7 = "llh.dim"(%arg0, %3) : (tensor<1x?x?x?xf32>, i64) -> i64
  %8 = "llh.dim"(%arg0, %1) : (tensor<1x?x?x?xf32>, i64) -> i64
  %9 = "llh.dim"(%arg0, %0) : (tensor<1x?x?x?xf32>, i64) -> i64
  // CHECK: mhlo.dynamic_broadcast_in_dim
  %10 = "llh.broadcast_to"(%5, %6, %7, %8, %9) <{cast_dims = array<i64: 1, 2, 3>}> : (tensor<1x1x1x1xf32>, i64, i64, i64, i64) -> tensor<1x?x?x?xf32>
  %11 = "llh.add"(%arg0, %10) : (tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  return %11 : tensor<1x?x?x?xf32>
}

