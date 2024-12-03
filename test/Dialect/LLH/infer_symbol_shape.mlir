// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --infer-symbol-shape /home/lfr/LLCompiler/test/Dialect/LLH/infer_symbol_shape.mlir


// CHECK: llh.symbol_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<GE>
// CHECK: llh.symbol_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<GE>
// CHECK: llh.symbol_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<GE>
// CHECK-LABEL: block
// CHECK-SAME: (%arg0: tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, %arg1: bf16) -> tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>
func.func @block(%arg2: tensor<?x3x?x?xbf16>,%arg3: bf16) ->(tensor<?x3x?x?xbf16>) attributes {entrance}{
  return %arg2 : tensor<?x3x?x?xbf16>
}

// -----
// CHECK: llh.symbolic_int
// CHECK-SAME: sym_name = "c384"
// CHECK-LABEL: constant
func.func @constant() ->() attributes {entrance}{
  // CHECK: llh.constant
  // CHECK-SAME: tensor<384xf32, #llh.encoding<shapes = @c384>>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<*xf32>
  return 
}

// -----
// CHECK: #map = affine_map<(d0)[s0] -> ((s0 - 1) ceildiv 2 + 1)>
// CHECK-LABEL: conv
func.func @conv(%arg0: tensor<?x3x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"} ) ->() attributes {entrance}{
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-29T23:48:46.139597+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>
  // CHECK: llh.conv
  // CHECK-SAME: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>
  %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<*xf32>
  return 
}

// -----
// CHECK-LABEL: conv_static
func.func @conv_static(%arg0: tensor<2x3x224x224xf32>) ->() attributes {entrance}{
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-29T23:48:46.139597+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32>
  // CHECK: llh.conv
  // CHECK-SAME:-> tensor<2x64x109x210xf32, #llh.encoding<shapes = @c2, @c64, @c109, @c210>>
  %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 2, 3>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 2, 3, 2>, stride = array<i64: 2, 1>}> : (tensor<2x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<*xf32>
  return 
}

// -----
// CHECK-LABEL: transpose
func.func @transpose(%arg0: tensor<?x3x?x?xf32>) -> () attributes {entrance} {
  // CHECK: llh.transpose
  // CHECK-SAME: tensor<?x?x3x?xf32, #llh.encoding<shapes = @s2, @s1, @c3, @s0>>
  %30 = "llh.transpose"(%arg0) <{perms = array<i64: 3, 2, 1, 0>}> : (tensor<?x3x?x?xf32>) -> tensor<*xf32>
  return 
}

// -----
// CHECK-LABEL: empty
func.func @empty(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?xf32>) attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %5 = "llh.dim"(%arg0, %4) : (tensor<?x?x?x?xf32>, i64) -> i64
  %6 = "llh.dim"(%arg0, %3) : (tensor<?x?x?x?xf32>, i64) -> i64
  %7 = "llh.dim"(%arg0, %2) : (tensor<?x?x?x?xf32>, i64) -> i64
  %8 = "llh.dim"(%arg0, %0) : (tensor<?x?x?x?xf32>, i64) -> i64
  // CHECK: llh.empty
  // CHECK-SAME: -> tensor<?x?xf32, #llh.encoding<shapes = @s2, @s1>>
  %22 = "llh.empty"(%6, %7) : (i64, i64) -> tensor<?x?xf32>
  return  %22: tensor<?x?xf32>
}

// -----
// CHECK-LABEL: max_pool
func.func @max_pool(%arg0: tensor<?x64x?x?xf32>) -> () attributes {entrance} {
 // CHECK: llh.max_pool
 // CHECK-SAME: -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s3, @s4>>
 %129 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<*xf32>
  return 
}