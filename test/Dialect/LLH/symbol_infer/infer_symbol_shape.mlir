// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s
// RUN: llc-opt --split-input-file -infer-symbol-shape="use-encoding=false " %s | FileCheck %s --check-prefix=CHECK-ENCODING
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --infer-symbol-shape /home/lfr/LLCompiler/test/Dialect/LLH/symbol_infer/infer_symbol_shape.mlir
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file -infer-symbol-shape="use-encoding=false " /home/lfr/LLCompiler/test/Dialect/LLH/symbol_infer/infer_symbol_shape.mlir

// CHECK-LABEL: block
// CHECK-SAME: (%arg0: tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, %arg1: bf16) -> tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>
// CHECK-ENCODING: llh.encoding_bind
func.func @block(%arg2: tensor<?x3x?x?xbf16>,%arg3: bf16) ->(tensor<?x3x?x?xbf16>) attributes {entrance}{
  return %arg2 : tensor<?x3x?x?xbf16>
}
// CHECK: llh.symbol_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<GE>
// CHECK: llh.symbol_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<GE>
// CHECK: llh.symbol_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<GE>

// -----
// CHECK: func.func
// CHECK-SAME: -> (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) 
// CHECK-ENCODING: llh.encoding_bind
func.func @checkIsReturnOperand(%arg0: tensor<?x?x?x?xf32>, %arg1: i64) -> (tensor<*xf32> , i64) attributes {entrance} {
  %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<*xf32>
  // CHECK: return
  // CHECK-SAME: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64
  return %0, %arg1 : tensor<*xf32>, i64
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK: llh.symbolic_int
// CHECK-SAME: sym_name = "c384"
// CHECK-LABEL: constant
func.func @constant() ->(tensor<*xf32>) attributes {entrance}{
  // CHECK: llh.constant
  // CHECK-SAME: tensor<384xf32, #llh.encoding<shapes = @c384>>
  %0 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<*xf32>
  return %0: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK-LABEL: transpose
func.func @transpose(%arg0: tensor<?x3x?x?xf32>) -> (tensor<*xf32>) attributes {entrance} {
  // CHECK: llh.transpose
  // CHECK-SAME: tensor<?x?x3x?xf32, #llh.encoding<shapes = @s2, @s1, @c3, @s0>>
  %0 = "llh.transpose"(%arg0) <{perms = array<i64: 3, 2, 1, 0>}> : (tensor<?x3x?x?xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK-LABEL: empty
func.func @empty(%arg0: tensor<?x?x?x?xf32>) -> (tensor<*xf32>) attributes {entrance} {
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
  %22 = "llh.empty"(%6, %7) : (i64, i64) -> tensor<*xf32>
  return %22: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK: #map = affine_map<(d0)[s0] -> ((s0 - 1) ceildiv 2 + 1)>
// CHECK-LABEL: max_pool
func.func @max_pool(%arg0: tensor<?x64x?x?xf32>) -> (tensor<*xf32>) attributes {entrance} {
  // CHECK: llh.max_pool
  // CHECK-SAME: -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s3, @s4>>
  %0 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
  // CHECK: llh.symbol_relation_map
  // CHECK: llh.symbol_relation_map
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK-LABEL: max_pool_static
func.func @max_pool_static(%arg0: tensor<2x64x9x17xf32>) -> (tensor<*xf32>) attributes {entrance} {
  // CHECK: llh.max_pool
  // CHECK-SAME: -> tensor<2x64x2x7xf32, #llh.encoding<shapes = @c2, @c64, @c2, @c7>>
  %0 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 2, 1>, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 1, 3, 1>, stride = array<i64: 2, 2>}> : (tensor<2x64x9x17xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @reshape(%arg0: tensor<?x?x224x226xf32>) ->(tensor<*xf32>) attributes {entrance}{
  %7 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 6 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %5 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %14 = "llh.dim"(%arg0, %7) : (tensor<?x?x224x226xf32, >, i64) -> i64
  %15 = "llh.dim"(%arg0, %5) : (tensor<?x?x224x226xf32, >, i64) -> i64
  // CHECK: llh.reshape
  // CHECK-SAME:-> tensor<1x6x224x226xf32, #llh.encoding<shapes = @c1, @c6, @c224, @c226>>
  %16 = "llh.reshape"(%arg0, %4, %3, %14, %15) : (tensor<?x?x224x226xf32, >, i64, i64, i64, i64) -> tensor<*xf32>
  return %16: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @reshape(%arg0: tensor<?x?x224x226xf32>) ->(tensor<*xf32>) attributes {entrance}{
  %7 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 6 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %5 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %14 = "llh.dim"(%arg0, %7) : (tensor<?x?x224x226xf32, >, i64) -> i64
  %15 = "llh.dim"(%arg0, %5) : (tensor<?x?x224x226xf32, >, i64) -> i64
  // CHECK: llh.reshape
  // CHECK-SAME:-> tensor<1x6x224x226xf32, #llh.encoding<shapes = @c1, @c6, @c224, @c226>>
  %16 = "llh.reshape"(%arg0, %4, %3, %14, %15) : (tensor<?x?x224x226xf32, >, i64, i64, i64, i64) -> tensor<*xf32>
  return %16: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK-LABEL: adaptive_average_pool
func.func @adaptive_average_pool(%arg0: tensor<2x64x9x17xf32>) -> (tensor<*xf32>) attributes {entrance} {
  // CHECK: llh.adaptive_average_pool
  // CHECK-SAME: -> tensor<2x64x1x1xf32, #llh.encoding<shapes = @c2, @c64, @c1, @c1>>
  %0 = "llh.adaptive_average_pool"(%arg0) <{out_size = array<i64: 1, 1>}> : (tensor<2x64x9x17xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK-LABEL: binary
func.func @binary(%arg0: tensor<?x3x?x?xf32>, %arg2: tensor<1x1x?x?xf32>) ->(tensor<*xf32>) attributes {entrance}{
  // CHECK: llh.add
  // CHECK-SAME: tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>
  %6 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<1x3xf32>
  %125 = "llh.add"(%6, %6): (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<*xf32>
  return %125: tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @matmul(%arg0: tensor<?x512xf32>) -> tensor<*xf32> attributes {entrance} {
    // CHECK: llh.matmul
    // CHECK-SAME: tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %const = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<512x10xf32>
    %matmul = "llh.matmul"(%arg0, %const) : (tensor<?x512xf32>, tensor<512x10xf32>) -> tensor<*xf32>
    // llh.symbol_relation
    return %matmul : tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK: #map = affine_map<(d0)[s0, s1] -> (s0 - s1)>
func.func @stride_slice(%arg0: tensor<?x?x?x?xf32>) -> (tensor<*xf32>) attributes {entrance} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3_i64 = arith.constant 3 : i64
    %0 = "llh.dim"(%arg0, %c1_i64) : (tensor<?x?x?x?xf32>, i64) -> i64
    %1 = "llh.dim"(%arg0, %c2_i64) : (tensor<?x?x?x?xf32>, i64) -> i64
    %2 = "llh.dim"(%arg0, %c3_i64) : (tensor<?x?x?x?xf32>, i64) -> i64
    // CHECK: llh.stride_slice
    // CHECK-SAME: tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s4, @s5, @s3>>
    %3 = "llh.stride_slice"(%arg0, %c0_i64, %c2_i64, %c3_i64, %c0_i64, %c1_i64, %0, %1, %2, %c1_i64, %c1_i64, %c1_i64, %c1_i64) <{operandSegmentSizes = array<i32: 1, 4, 4, 4>}> : (tensor<?x?x?x?xf32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tensor<*xf32>
    return %3 : tensor<*xf32>

  // CHECK: llh.symbol_relation_map
  // CHECK-SMAE: express = "-3 + s2"
  // CHECK: llh.symbol_relation_map
  // CHECK-SMAE: express = "-2 + s1"
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK: #map = affine_map<(d0)[s0] -> ((s0 - 1) ceildiv 2 + 1)>
// CHECK-LABEL: conv
func.func @conv(%arg0: tensor<?x3x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"} ) ->(tensor<*xf32>) attributes {entrance}{
  %0 = "llh.weight"() <{weight_file = "npy"}> : () -> tensor<64x3x7x7xf32>
  // CHECK: llh.conv
  // CHECK-SAME: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s1>>
  %1 = "llh.conv"(%arg0, %0) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<64x3x7x7xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
  // CHECK: llh.symbol_relation_map
  // CHECK-SMAE: express = "(1.0/2.0)*(1 + s2)"
}

// -----
// CHECK-ENCODING: llh.encoding_bind
// CHECK-LABEL: conv_static
func.func @conv_static(%arg0: tensor<2x3x224x224xf32>) ->(tensor<*xf32>) attributes {entrance}{
  %0 = "llh.weight"() <{weight_file = "npy"}> : () -> tensor<64x3x7x7xf32>
  // CHECK: llh.conv
  // CHECK-SAME:-> tensor<2x64x109x210xf32, #llh.encoding<shapes = @c2, @c64, @c109, @c210>>
  %1 = "llh.conv"(%arg0, %0) <{dilation = array<i64: 2, 3>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 2, 3, 2>, stride = array<i64: 2, 1>}> : (tensor<2x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @broadcast_to(%arg0: tensor<?x?x?x?xf32>) ->(tensor<*xf32>) attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %3 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %5 = "llh.mul"(%arg0, %arg0) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  %6 = "llh.reshape"(%2, %1, %1, %1, %1) : (tensor<1xf32>, i64, i64, i64, i64) -> tensor<*xf32>
  %7 = "llh.dim"(%5, %4) : (tensor<?x?x?x?xf32>, i64) -> i64
  %8 = "llh.dim"(%5, %1) : (tensor<?x?x?x?xf32>, i64) -> i64
  %9 = "llh.dim"(%5, %3) : (tensor<?x?x?x?xf32>, i64) -> i64
  %10 = "llh.dim"(%5, %0) : (tensor<?x?x?x?xf32>, i64) -> i64
  // CHECK: llh.broadcast_to
  // CHECK-SAME:-> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
  %11 = "llh.broadcast_to"(%6, %7, %8, %9, %10) <{cast_dims = array<i64: 0, 1, 2, 3>}> : (tensor<*xf32>, i64, i64, i64, i64) -> tensor<*xf32>
  return %11 : tensor<*xf32>
}


// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @batch_matmul(%arg0: tensor<12x?x512xf32>) -> tensor<*xf32> attributes {entrance} {
    // CHECK: llh.batch_matmul
    // CHECK-SAME:  tensor<12x?x10xf32, #llh.encoding<shapes = @c12, @s0, @c10>>
    %const = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<12x512x10xf32>
    %matmul = "llh.batch_matmul"(%arg0, %const) : (tensor<12x?x512xf32>, tensor<12x512x10xf32>) -> tensor<*xf32>
    // llh.symbol_relation
    return %matmul : tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @batch_norm(%arg0: tensor<3xf32> {func.input_symbol_0 = "c3"}, %arg1: tensor<3xf32> {func.input_symbol_0 = "c3"}, %arg2: tensor<3xf32> {func.input_symbol_0 = "c3"}, %arg3: tensor<3xf32> {func.input_symbol_0 = "c3"}, %arg4: tensor<1xi64> {func.input_symbol_0 = "c1"}, %arg5: tensor<?x3x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "c3", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}) -> (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<?x3x?x?xf32>, tensor<0xf32>, tensor<0xf32>) attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    // CHECK: llh.batch_norm
    // CHECK-SAME: : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, tensor<3xf32, #llh.encoding<shapes = @c3>>, tensor<3xf32, #llh.encoding<shapes = @c3>>, tensor<3xf32, #llh.encoding<shapes = @c3>>, tensor<3xf32, #llh.encoding<shapes = @c3>>) -> (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s2, @s2>>, tensor<0xf32, #llh.encoding<shapes = @c0>>, tensor<0xf32, #llh.encoding<shapes = @c0>>)
    %result, %running_mean, %running_var = "llh.batch_norm"(%arg5, %arg0, %arg1, %arg2, %arg3) <{epsilon = 1.000000e-01 : f64, feature_index = 1 : i64, mode = #llh.Mode<inference>, momentum = 1.000000e-05 : f64}> : (tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<?x3x?x?xf32>, tensor<0xf32>, tensor<0xf32>)
    return %result, %arg0, %arg2, %arg3, %arg5, %running_mean, %running_var : tensor<?x3x?x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<?x3x?x?xf32>, tensor<0xf32>, tensor<0xf32>
  }

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @extract(%arg0: tensor<?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2"}) -> tensor<*xf32> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c18446744073709551615, value = -1 : i64}> : () -> i64
    // CHECK: llh.extract
    // CHECK-SAME:-> tensor<?x?xf32, #llh.encoding<shapes = @s1, @s2>>
    %5 = "llh.extract"(%arg0, %3) : (tensor<?x?x?xf32>, i64) -> tensor<*xf32>
    // CHECK: llh.extract
    // CHECK-SAME:-> tensor<?xf32, #llh.encoding<shapes = @s2>>
    %6 = "llh.extract"(%5, %4) : (tensor<*xf32>, i64) -> tensor<*xf32>
    // CHECK: llh.extract
    // CHECK-SAME:-> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %7 = "llh.extract"(%6, %3) : (tensor<*xf32>, i64) -> tensor<*xf32>
    // CHECK: llh.extract
    // CHECK-SAME:-> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %8 = "llh.extract"(%7, %3) : (tensor<*xf32>, i64) -> tensor<*xf32>
    return %8 : tensor<*xf32>
}

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @where(%arg0: tensor<?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}, %arg1: tensor<?x?xi1> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}, %arg2: tensor<?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}) -> tensor<*xf32> attributes {entrance} {
    // CHECK: llh.where
    // CHECK-SAME:-> tensor<?x?xf32, #llh.encoding<shapes = @s0, @s1>>
    %2 = "llh.where"(%arg1, %arg0, %arg0) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
    return %2 : tensor<*xf32>
  }

// -----
// CHECK-ENCODING: llh.encoding_bind
func.func @reduce(%arg0: tensor<3x3x?x?xf32> {func.input_symbol_0 = "c3", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<3x3x1x?xf32> attributes {entrance} {
    %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
    // CHECK: llh.reduce_max
    // CHECK-SAME:-> tensor<3x3x1x?xf32, #llh.encoding<shapes = @c3, @c3, @c1, @s1>>
    %1 = "llh.reduce_max"(%arg0) <{axis = 2 : i64}> : (tensor<3x3x?x?xf32>) -> tensor<3x3x1x?xf32>
    return %1 : tensor<3x3x1x?xf32>
  }