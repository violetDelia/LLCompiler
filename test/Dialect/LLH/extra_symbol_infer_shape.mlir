// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s
// RUN: llc-opt --split-input-file -infer-symbol-shape="use-encoding=false " %s | FileCheck %s --check-prefix=CHECK-ENCODING
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --infer-symbol-shape /home/lfr/LLCompiler/test/Dialect/LLH/extra_symbol_infer_shape.mlir
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file -infer-symbol-shape="use-encoding=false " /home/lfr/LLCompiler/test/Dialect/LLH/extra_symbol_infer_shape.mlir

// CHECK-LABEL: const
func.func @const(%arg0: tensor<?x512x1x1xf32>) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c3
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c2
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c0
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c1
  return 
}

// -----
// CHECK: #map = affine_map<(d0)[s0, s1] -> (s0 * s1)>
// CHECK-LABEL: mul
func.func @mul(%arg0: tensor<?x512x1x1xf32>) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %193 = "llh.dim"(%arg0, %2) : (tensor<?x512x1x1xf32>, i64) -> i64
  %194 = "llh.dim"(%arg0, %3) : (tensor<?x512x1x1xf32>, i64) -> i64
  %195 = "llh.dim"(%arg0, %1) : (tensor<?x512x1x1xf32>, i64) -> i64
  %196 = "llh.dim"(%arg0, %0) : (tensor<?x512x1x1xf32>, i64) -> i64
  // CHECK: llh.mul
  // CHECK-SAME: symbol = @s1
  // CHECK: llh.mul
  // CHECK-SAME: symbol = @c512
  // CHECK: llh.mul
  // CHECK-SAME: symbol = @c1
  %197 = "llh.mul"(%193, %194) : (i64, i64) -> i64
  %198 = "llh.mul"(%194, %195) : (i64, i64) -> i64
  %199 = "llh.mul"(%196, %195) : (i64, i64) -> i64
  return 
}
// CHECK-LABEL: __symbol__
// CHECK: llh.symbol_relation_map
// CHECK-SAME: express = "512*s0", relation = #map, relations = [@s0, @c512], symbol = @s1

// -----
// CHECK-ENCODING-LABEL: arith_const
func.func @arith_const() -> (index) attributes {entrance} {
  // CHECK-ENCODING: arith.constant
  // CHECK-ENCODING: symbol = @c1
  %c1 = arith.constant 1 : index
  return %c1: index
}

// -----
// CHECK-ENCODING-LABEL: tensor_dim
func.func @tensor_dim(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> (index) attributes {entrance} {
  // CHECK-ENCODING: encoding_bind
  // CHECK-ENCODING-SAME: shapes = @s0, @s0, @s1, @s1
  %c2 = arith.constant 2 : index
  // CHECK-ENCODING: tensor.dim
  // CHECK-ENCODING-SAME: symbol = @s1
  %dim_0 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
  return %dim_0: index
}