// RUN: llc-opt --split-input-file --inline %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --inline /home/lfr/LLCompiler/test/Dialect/LLH/fold.mlir

// CHECK-LABEL: fold_dim
func.func @fold_dim(%101: tensor<?x512x1x1xf32>) ->(i64, i64, i64, i64) attributes {entrance} {
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
// CHECK-LABEL: add_zore_fold
func.func @add_zore_fold(%arg2: i64) ->(i64) attributes {entrance} {
  %0 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  // CHECK-NOT: llh.add
  %res = "llh.add"(%0, %arg2) : (i64, i64) -> i64
  return %res: i64
}

// -----
// CHECK-LABEL: add_zore_fold
func.func @add_zore_fold(%arg2: i64) ->(i64) attributes {entrance} {
  %0 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  // CHECK-NOT: llh.add
  %res = "llh.add"(%arg2, %0) : (i64, i64) -> i64
  return %res: i64
}

// -----
// CHECK-LABEL: add_sub_fold
func.func @add_sub_fold(%arg2: f32, %arg3: f32) ->(f32) attributes {entrance} {
  %103 = "llh.sub"(%arg2, %arg3) : (f32, f32) -> f32
  // CHECK-NOT: llh.add
  %res = "llh.add"(%103, %arg3) : (f32, f32) -> f32
  return %res : f32
}

// -----
// CHECK-LABEL: add_sub_fold
func.func @add_sub_fold(%arg2: f32, %arg3: f32) ->(f32) attributes {entrance} {
  %103 = "llh.sub"(%arg2, %arg3) : (f32, f32) -> f32
  // CHECK-NOT: llh.add
  %res = "llh.add"(%arg3, %103) : (f32, f32) -> f32
  return %res : f32
}

// -----
// CHECK-LABEL: add_zore_fold
func.func @add_zore_fold(%arg0: tensor<64xf32>) ->(tensor<64xf32>) attributes {entrance} {
  %7 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  // CHECK-NOT: llh.add
  %res = "llh.add"(%arg0, %7) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %res: tensor<64xf32>
}

// -----
// CHECK-LABEL: sub_same_fold
func.func @sub_same_fold(%arg0: tensor<64xf32>) ->(tensor<64xf32>) attributes {entrance} {
  %7 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %8 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  // CHECK-NOT: llh.sub
  %res = "llh.sub"(%7, %7) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %c = "llh.add"(%res, %7) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %c: tensor<64xf32>
}

// -----
// CHECK-LABEL: sub_add_fold
func.func @sub_add_fold() ->(i64, i64) attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  // CHECK-NOT: llh.sub
  %add = "llh.add"(%2, %3): (i64, i64) -> i64
  %sub = "llh.sub"(%add,%2) :(i64, i64) -> i64
  %sub2 = "llh.sub"(%add,%2) :(i64, i64) -> i64
  return %sub, %sub2 : i64, i64
}

// -----
// CHECK-LABEL: div_fold
func.func @div_fold() ->(i64, i64,tensor<64xf32>) attributes {entrance} {
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %f2 = "llh.constant"() <{value = 2. : f32}> : () -> f32
  %f1 = "llh.constant"() <{value = 1. : f32}> : () -> f32
  %7 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %8 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %9 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  // CHECK-COUNT-1: llh.div
  %div = "llh.div"(%2, %0): (i64, i64) -> i64
  %div_f = "llh.div"(%f2, %f1): (f32, f32) -> f32
  %div_tensor = "llh.div"(%8, %7): (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %div_zore = "llh.div"(%2, %3): (i64, i64) -> i64
  return %div, %div_zore,%div_tensor : i64, i64,tensor<64xf32>
}

// -----
// CHECK-LABEL: mul_fold
func.func @mul_fold() ->(i64, f32, tensor<64xf32>, tensor<64xf32>) attributes {entrance} {
  %i1 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %i2 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %f2 = "llh.constant"() <{value = 2. : f32}> : () -> f32
  %f1 = "llh.constant"() <{value = 1. : f32}> : () -> f32
  %tensor_1 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %tensor = "llh.constant"() <{value = dense<2.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %tensor_0 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  // CHECK-NOT: llh.mul
  %mul_0 = "llh.mul"(%i2, %i1): (i64, i64) -> i64
  %mul_1 = "llh.mul"(%f2, %f1): (f32, f32) -> f32
  %mul_tensor_0 = "llh.mul"(%tensor, %tensor_0): (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %mul_tensor_1 = "llh.mul"(%tensor, %tensor_1): (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %mul_0, %mul_1,%mul_tensor_1,%mul_tensor_0 : i64, f32, tensor<64xf32>, tensor<64xf32>
}
