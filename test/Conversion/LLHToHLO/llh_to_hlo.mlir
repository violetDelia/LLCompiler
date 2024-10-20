// RUN: llc-opt --split-input-file --convert-llh-to-hlo %s| FileCheck %s

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-hlo /home/lfr/LLCompiler/test/Conversion/LLHToHLO/llh_to_hlo.mlir

func.func @constant() ->() attributes {entrance}{
  // CHECK: stablehlo.constant
  // CHECK-SAME: tensor<384xf32>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  // CHECK: llh.constant
  // CHECK-SAME: -> i64
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}

// -----

func.func @add_and_sub(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: stablehlo.subtract
  %23 = "llh.sub"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  // CHECK: stablehlo.add
  %24 = "llh.add"(%23, %23) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----

func.func @mul(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: stablehlo.multiply
  %23 = "llh.mul"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----

func.func @div(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  %98 = "llh.constant"() <{value = dense<1> : tensor<384xi32>}> : () -> tensor<384xi32>
  // CHECK: stablehlo.div
  %222 = "llh.div"(%98, %98) : (tensor<384xi32>, tensor<384xi32>) -> tensor<384xi32>
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
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %10 = "llh.broadcast_to"(%5, %6, %7, %8, %9) <{cast_dims = array<i64: 1, 2, 3>}> : (tensor<1x1x1x1xf32>, i64, i64, i64, i64) -> tensor<1x?x?x?xf32>
  %11 = "llh.add"(%arg0, %10) : (tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  return %11 : tensor<1x?x?x?xf32>
}

// -----

func.func @relu(%arg0: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> attributes {entrance} {
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
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  %10 = "llh.broadcast_to"(%5, %6, %7, %8, %9) <{cast_dims = array<i64: 1, 2, 3>}> : (tensor<1x1x1x1xf32>, i64, i64, i64, i64) -> tensor<1x?x?x?xf32>
  %11 = "llh.add"(%arg0, %10) : (tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  return %11 : tensor<1x?x?x?xf32>
}