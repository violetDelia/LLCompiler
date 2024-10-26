// RUN: llc-opt --split-input-file --convert-llh-to-tosa %s| FileCheck %s

func.func @constant() ->() attributes {entrance}{
  // CHECK: tosa.const
  // CHECK-SAME: tensor<384xf32>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  // CHECK: llh.constant
  // CHECK-SAME: -> i64
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}

// -----
func.func @add_and_sub(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: tosa.sub
  %23 = "llh.sub"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  // CHECK: tosa.add
  %24 = "llh.add"(%23, %23) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----
func.func @mul(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: tosa.mul
  %23 = "llh.mul"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----
func.func @div(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  %23 = "llh.div"(%arg0, %arg0) : (tensor<?x10x?x?xf32>, tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----
func.func @conv(%arg0: tensor<200x96x96x3xf32>,%arg1: tensor<3x7x7x3xf32>) ->() attributes {entrance}{
  // CHECK: tosa.const
  // CHECK: tosa.conv2d
  %6 = "llh.conv"(%arg0, %arg1) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NHWC>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<200x96x96x3xf32>, tensor<3x7x7x3xf32>) -> tensor<200x92x92x3xf32>
  return 
}