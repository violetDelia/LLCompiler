// RUN: llc-opt --split-input-file --convert-llh-to-hlo %s| FileCheck %s

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
  // CHECK: stablehlo.divide
  %222 = "llh.div"(%98, %98) : (tensor<384xi32>, tensor<384xi32>) -> tensor<384xi32>
  return 
}

// -----
func.func @abs(%arg0: tensor<?x10x?x?xf32>) ->() attributes {entrance}{
  // CHECK: stablehlo.abs
  %222 = "llh.abs"(%arg0) : (tensor<?x10x?x?xf32>) -> tensor<?x10x?x?xf32>
  return 
}

// -----
func.func @conv_nchw_fchw(%arg0: tensor<4x3x5x5xf32> , %arg1: tensor<?x3x?x?xf32>) -> () attributes {entrance} {
    // CHECK: stablehlo.convolution
    // CHECK-SAME: dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
    %2 = "llh.conv"(%arg1, %arg0) <{dilation = array<i64: 2, 2>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, layout = #llh.Layout<NCHW>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<4x3x5x5xf32>) -> tensor<?x4x?x?xf32>
    // CHECK: stablehlo.convolution
    // dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
    %3 = "llh.conv"(%arg1, %arg0) <{dilation = array<i64: 2, 2>, group = 1 : i64, kernel_shape = array<i64: 5, 5>, layout = #llh.Layout<NHWC>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32>, tensor<4x3x5x5xf32>) -> tensor<?x2x?x4xf32>
    return 
}

// -----
func.func @transpose(%arg0: tensor<?x?x?x4xf32>) -> () attributes {entrance} {
    // CHECK-NOT: llh.transpose
    // CHECK: stablehlo.transpose
    // CHECK-SAME: (tensor<?x?x?x4xf32>) -> tensor<?x4x?x?xf32>
    %7 = "llh.transpose"(%arg0) <{perms = array<i64: 0, 3, 1, 2>}> : (tensor<?x?x?x4xf32>) -> tensor<?x4x?x?xf32>
    return 
}


// -----
func.func @batch_norm(%arg0: tensor<3x3x100x100xf32> ) -> tensor<3x3x100x100xf32> attributes {entrance} {
    %1 = "llh.constant"() <{value = dense<[0.866779148, 0.87528336, 0.868859171]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %2 = "llh.constant"() <{value = dense<[7.19547679E-5, 6.82539321E-5, 1.0772681E-4]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %3 = "llh.constant"() <{value = dense<0.000000e+00> : tensor<3xf32>}> : () -> tensor<3xf32>
    %4 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<3xf32>}> : () -> tensor<3xf32>
    // CHECK: stablehlo.batch_norm_inference
    // CHECK-NOT: llh.batch_norm
    %7 = "llh.batch_norm_inference"(%arg0, %4, %3, %2, %1) <{epsilon = 1.23 : f64, feature_index = 1 : i64, momentum = 1.23 : f64}> : (tensor<3x3x100x100xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3x3x100x100xf32>
    return %7 : tensor<3x3x100x100xf32>
}

// -----
func.func @matmul(%arg0: tensor<?x100xf32> ) -> tensor<?x10xf32> attributes {entrance} {
    %0 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<10x100xf32>
    %c0_i64 = arith.constant {symbol = @c0} 0 : i64
    %1 = "llh.transpose"(%0) <{perms = array<i64: 1, 0>}> : (tensor<10x100xf32>) -> tensor<100x10xf32>
    // CHECK: stablehlo.dot
    // CHECK-NOT: llh.matmul
    %2 = "llh.matmul"(%arg0, %1) : (tensor<?x100xf32>, tensor<100x10xf32>) -> tensor<?x10xf32>
    return %2 : tensor<?x10xf32>
}

// -----
func.func @max_pool(%arg0: tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> attributes {entrance} {
    // CHECK: stablehlo.constant
    // CHECK: stablehlo.reduce_window
    // CHECK: stablehlo.maximum
    // CHECK: stablehlo.return
    // CHECK-NOT: llh.max_pool
    %0 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 2, 1>, kernel_shape = array<i64: 5, 5>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 1, 2>}> : (tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
    return %0 : tensor<1x?x?x?xf32>
}

// -----
func.func @slice(%arg0: tensor<?x?x?x?xf32> ) -> tensor<1x?x?x?xf32> attributes {entrance} {
    %c0_i64 = arith.constant {symbol = @c0} 0 : i64
    %c1_i64 = arith.constant {symbol = @c1} 1 : i64
    %c2_i64 = arith.constant {symbol = @c2} 2 : i64
    %c3_i64 = arith.constant {symbol = @c3} 3 : i64
    %0 = "llh.dim"(%arg0, %c1_i64) <{symbol = @s1}> : (tensor<?x?x?x?xf32>, i64) -> i64
    %1 = "llh.dim"(%arg0, %c2_i64) <{symbol = @s2}> : (tensor<?x?x?x?xf32>, i64) -> i64
    %2 = "llh.dim"(%arg0, %c3_i64) <{symbol = @s3}> : (tensor<?x?x?x?xf32>, i64) -> i64
    // CHECK: stablehlo.real_dynamic_slice
    // CHECK-NOT: llh.stride_slice
    %3 = "llh.stride_slice"(%arg0, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %0, %1, %2, %c1_i64, %c1_i64, %c1_i64, %c1_i64) <{operandSegmentSizes = array<i32: 1, 4, 4, 4>}> : (tensor<?x?x?x?xf32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tensor<1x?x?x?xf32>
    return %3 : tensor<1x?x?x?xf32>
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
  // CHECK-NOT: llh.broadcast_to
  %10 = "llh.broadcast_to"(%5, %6, %7, %8, %9) <{cast_dims = array<i64: 0, 1, 2, 3>}> : (tensor<1x1x1x1xf32>, i64, i64, i64, i64) -> tensor<1x?x?x?xf32>
  %11 = "llh.add"(%arg0, %10) : (tensor<1x?x?x?xf32>, tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  return %11 : tensor<1x?x?x?xf32>
}

// -----
func.func @batch_matmul(%arg0: tensor<12x?x512xf32>) -> tensor<12x?x10xf32> attributes {entrance} {
    // CHECK-NOT: llh.batch_matmul
    %const = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<12x512x10xf32>
    %matmul = "llh.batch_matmul"(%arg0, %const) : (tensor<12x?x512xf32>, tensor<12x512x10xf32>) -> tensor<12x?x10xf32>
    return %matmul : tensor<12x?x10xf32>
}

// -----
func.func @compare(%arg0: tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xi1>) attributes {entrance} {
    // CHECK-NOT: llh.compare
    // CHECK: stablehlo.compare
    // CHECK-SAME: EQ, %arg0, %arg0,  FLOAT
    // CHECK: stablehlo.compare
    // CHECK-SAME: GE, %arg0, %arg0,  FLOAT
    // CHECK: stablehlo.compare
    // CHECK-SAME: SIGNED
    %f_eq = "llh.compare"(%arg0, %arg0) <{kind = #llh.Compare<EQ>}> : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
    %f_ge = "llh.compare"(%arg0, %arg0) <{kind = #llh.Compare<GE>}> : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xi1>
    %s_le = "llh.compare"(%f_eq, %f_ge) <{kind = #llh.Compare<LE>}> : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xi1>) -> tensor<?x?x?x?xi1>
    return %s_le: tensor<?x?x?x?xi1>
}

// -----
func.func @where(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi1>) -> tensor<?x?xf32> attributes {entrance} {
    // CHECK-NOT: llh.where
    // CHECK: stablehlo.select
    %2 = "llh.where"(%arg1, %arg0, %arg0) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }

// -----
func.func @convert_to(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xi1>) -> tensor<?x?x?xi1> attributes {entrance} {
    // CHECK-NOT: llh.convert_to
    // CHECK: stablehlo.convert
    %0 = "llh.convert_to"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xi1>
    return %0 : tensor<?x?x?xi1>
}

// -----
func.func @reduce_max(%arg0: tensor<3x3x?x?xf32> {func.input_symbol_0 = "c3", func.input_symbol_1 = "c3", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<3x?xf32> attributes {entrance} {
  // CHECK-NOT: llh.reduce_max
  // stablehlo.reduce
  %0 = "llh.reduce_max"(%arg0) <{axis = array<i64:1, 2>}> : (tensor<3x3x?x?xf32>) -> tensor<3x?xf32>
  return %0 : tensor<3x?xf32>
}
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-llh-to-tensor --convert-llh-to-hlo --fold-index-cast --canonicalize /home/lfr/LLCompiler/test/Conversion/LLHToHLO/llh_to_hlo.mlir