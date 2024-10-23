// RUN: llc-opt --split-input-file --transform-layout-to-nhwc %s| FileCheck %s
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --transform-layout-to-nhwc /home/lfr/LLCompiler/test/Dialect/LLH/transform_layout_to_nhwc.mlir


"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c0"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c4"}> : () -> ()
// CHECK-LABEL: conv_nchw_to_nhwc
func.func @conv_nchw_to_nhwc(%arg0: tensor<4x3x7x7xf32, #llh.encoding<shapes = @c4, @c3, @c7, @c7>> , %arg1: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>) -> (tensor<?x4x?x?xf32, #llh.encoding<shapes = @s0, @c4, @s1, @s3>>, tensor<4x3x7x7xf32, #llh.encoding<shapes = @c4, @c3, @c7, @c7>>, tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, i64, i64) attributes {entrance} {
  %0 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
  %1 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
  %2 = "llh.dim"(%arg1, %1) <{symbol = @s0}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, i64) -> i64
  %3 = "llh.dim"(%arg1, %0) <{symbol = @s2}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, i64) -> i64
  // CHECK: llh.transpose
  // CHECK-SAME: perms = array<i64: 0, 2, 3, 1>
  // CHECK: llh.transpose
  // CHECK-SAME: perms = array<i64: 0, 2, 3, 1>
  // CHECK: llh.conv
  // CHECK: llh.transpose
  // CHECK-SAME: perms = array<i64: 0, 3, 1, 2>
  %4 = "llh.conv"(%arg1, %arg0) <{dilation = array<i64: 2, 2>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, tensor<4x3x7x7xf32, #llh.encoding<shapes = @c4, @c3, @c7, @c7>>) -> tensor<?x4x?x?xf32, #llh.encoding<shapes = @s0, @c4, @s1, @s3>>
  return %4, %arg0, %arg1, %2, %3 : tensor<?x4x?x?xf32, #llh.encoding<shapes = @s0, @c4, @s1, @s3>>, tensor<4x3x7x7xf32, #llh.encoding<shapes = @c4, @c3, @c7, @c7>>, tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, i64, i64
}
module @__symbol__ {
}

