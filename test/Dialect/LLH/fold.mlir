// RUN: llc-opt --split-input-file --inline %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --inline /home/lfr/LLCompiler/test/Dialect/LLH/fold.mlir

// CHECK-LABEL: fold_dim
func.func @fold_dim(%101: tensor<?x512x1x1xf32>) ->(i64, i64, i64, i64) attributes {entrance} {
  // CHECK-COUNT-5: llh.constant
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
func.func @add_zore_fold(%arg0: tensor<64xf32, #llh.encoding<shapes = @c64>>) ->(tensor<64xf32, #llh.encoding<shapes = @c64>>) attributes {entrance} {
  %7 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32, #llh.encoding<shapes = @c64>>
  // CHECK-NOT: llh.add
  %res = "llh.add"(%arg0, %7) : (tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<64xf32, #llh.encoding<shapes = @c64>>
  return %res: tensor<64xf32, #llh.encoding<shapes = @c64>>
}

// -----
// CHECK-LABEL: sub_same_fold
func.func @sub_same_fold(%arg0: tensor<64xf32, #llh.encoding<shapes = @c64>>) ->(tensor<64xf32, #llh.encoding<shapes = @c64>>) attributes {entrance} {
  %7 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32, #llh.encoding<shapes = @c64>>
  %8 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32, #llh.encoding<shapes = @c64>>
  // CHECK-NOT: llh.sub
  %res = "llh.sub"(%7, %7) : (tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<64xf32, #llh.encoding<shapes = @c64>>
  %c = "llh.add"(%res, %7) : (tensor<64xf32, #llh.encoding<shapes = @c64>>, tensor<64xf32, #llh.encoding<shapes = @c64>>) -> tensor<64xf32, #llh.encoding<shapes = @c64>>
  return %c: tensor<64xf32, #llh.encoding<shapes = @c64>>
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
