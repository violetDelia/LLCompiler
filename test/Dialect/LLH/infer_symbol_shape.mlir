// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --infer-symbol-shape /home/lfr/LLCompiler/test/Dialect/LLH/infer_symbol_shape.mlir

// CHECK: "llh.symbolic_int"() <{sym_name = "c384"}> : () -> ()
// CHECK-LABEL: constant
func.func @constant() ->() attributes {entrance}{
  // CHECK: llh.constant
  // CHECK-SAME: tensor<384xf32, #llh.encoding<shapes = @c384>>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<*xf32>
  return 
}

// -----
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: conv
func.func @conv(%arg0: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>) ->() attributes {entrance}{
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-29T23:48:46.139597+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>
  // CHECK: llh.conv
  // CHECK-SAME:-> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s3, @s4>>
  %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<*xf32>
  return 
}


// -----
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c7"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c224"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
// CHECK-LABEL: conv_static
func.func @conv_static(%arg0: tensor<2x3x224x224xf32, #llh.encoding<shapes = @c2, @c3, @c224, @c224>>) ->() attributes {entrance}{
  %4 = "llh.weight"() <{weight_file = "/home/lfr/LLCompiler/llcompiler/importer/LLcompiler_weight_temp/2024-09-29T23:48:46.139597+08:00/L__self___conv1.weight.npy"}> : () -> tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>
  // CHECK: llh.conv
  // CHECK-SAME:-> tensor<2x64x109x210xf32, #llh.encoding<shapes = @c2, @c64, @c109, @c210>>
  %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 2, 3>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 2, 3, 2>, stride = array<i64: 2, 1>}> : (tensor<2x3x224x224xf32, #llh.encoding<shapes = @c2, @c3, @c224, @c224>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<*xf32>
  return 
}


// -----
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: transpose
func.func @transpose(%arg0: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>) -> () attributes {entrance} {
  // CHECK: llh.transpose
  // CHECK-SAME: tensor<?x?x3x?xf32, #llh.encoding<shapes = @s2, @s1, @c3, @s0>>
  %30 = "llh.transpose"(%arg0) <{perms = array<i64: 3, 2, 1, 0>}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>) -> tensor<?x?x3x?xf32>
  return 
}

// -----
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: empty
func.func @empty(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> (tensor<?x?xf32>) attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %5 = "llh.dim"(%arg0, %4) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) -> i64
  %6 = "llh.dim"(%arg0, %3) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) -> i64
  %7 = "llh.dim"(%arg0, %2) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) -> i64
  %8 = "llh.dim"(%arg0, %0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) -> i64
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



// -----
"llh.symbolic_int"() <{sym_name = "c226"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c224"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
func.func @reshape(%arg0: tensor<?x?x224x226xf32, #llh.encoding<shapes = @s0, @s1, @c224, @c226>>) ->() attributes {entrance}{
  %7 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 6 : i64}> : () -> i64
  %4 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %5 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %14 = "llh.dim"(%arg0, %7) : (tensor<?x?x224x226xf32, #llh.encoding<shapes = @s0, @s1, @c224, @c226>>, i64) -> i64
  %15 = "llh.dim"(%arg0, %5) : (tensor<?x?x224x226xf32, #llh.encoding<shapes = @s0, @s1, @c224, @c226>>, i64) -> i64
  // CHECK: llh.reshape
  // CHECK-SAME:-> tensor<1x6x224x226xf32, #llh.encoding<shapes = @c1, @c6, @c224, @c226>>
  %16 = "llh.reshape"(%arg0, %4, %3, %14, %15) : (tensor<?x?x224x226xf32, #llh.encoding<shapes = @s0, @s1, @c224, @c226>>, i64, i64, i64, i64) -> tensor<*xf32>
  return 
}

// -----
"llh.symbolic_int"() <{sym_name = "c9"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c17"}> : () -> ()

// CHECK-LABEL: max_pool_static
func.func @max_pool_static(%arg0: tensor<2x64x9x17xf32, #llh.encoding<shapes = @c2, @c64, @c9, @c17>>) -> () attributes {entrance} {
 // CHECK: llh.max_pool
 // CHECK-SAME: -> tensor<2x64x2x7xf32, #llh.encoding<shapes = @c2, @c64, @c2, @c7>>
 %129 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 2, 1>, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 1, 3, 1>, stride = array<i64: 2, 2>}> : (tensor<2x64x9x17xf32, #llh.encoding<shapes = @c2, @c64, @c9, @c17>>) -> tensor<?x?x?x?xf32>
  return 
}

// -----
// CHECK: func.func
// CHECK-SAME: -> (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) 
func.func @checkIsReturnOperand(%arg0: tensor<?x?x?x?xf32>, %arg1: i64) -> (tensor<?x?x?x?xf32> , i64) attributes {entrance} {
  %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK: return
  // CHECK-SAME: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64
  return %0, %arg1 : tensor<?x?x?x?xf32>, i64
}

// -----

"llh.symbolic_int"() <{sym_name = "c9"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c17"}> : () -> ()

// CHECK-LABEL: adaptive_average_pool
func.func @adaptive_average_pool(%arg0: tensor<2x64x9x17xf32, #llh.encoding<shapes = @c2, @c64, @c9, @c17>>) -> () attributes {entrance} {
 // CHECK: llh.adaptive_average_pool
 // CHECK-SAME: -> tensor<2x64x1x1xf32, #llh.encoding<shapes = @c2, @c64, @c1, @c1>>
  %192 = "llh.adaptive_average_pool"(%arg0) <{out_size = array<i64: 1, 1>}> : (tensor<2x64x9x17xf32, #llh.encoding<shapes = @c2, @c64, @c9, @c17>>) -> tensor<*xf32>
  return 
}

// -----
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: binary
func.func @binary(%arg0: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, %arg2: tensor<1x1x?x?xf32, #llh.encoding<shapes = @c1, @c1, @s1, @s3>>) ->() attributes {entrance}{
  %4 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
  // CHECK: llh.add
  // CHECK-SAME: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>> 
  %126 = "llh.add"(%arg0, %4): (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<?x3x?x?xf32>
  // CHECK: llh.sub
  // CHECK-SAME: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>
  %127 = "llh.sub"(%arg0, %arg2): (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, tensor<1x1x?x?xf32, #llh.encoding<shapes = @c1, @c1, @s1, @s3>>) -> tensor<?x?x?x?xf32>

  // CHECK: llh.add
  // CHECK-SAME: tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>
  %6 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>
  %125 = "llh.add"(%6, %6): (tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>, tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>) -> tensor<*xf32>
  return 
}

// -----
func.func @broadcast_to(%arg0: tensor<?x?x?x?xf32>) ->() attributes {entrance} {
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
  return 
}

// -----
func.func @extract(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}) -> tensor<?x?xf32> attributes {entrance} {
    %0 = "llh.constant"() <{symbol = @c3, value = 3 : i64}> : () -> i64
    %1 = "llh.constant"() <{symbol = @c2, value = 2 : i64}> : () -> i64
    %2 = "llh.constant"() <{symbol = @c1, value = 1 : i64}> : () -> i64
    %3 = "llh.constant"() <{symbol = @c0, value = 0 : i64}> : () -> i64
    %4 = "llh.constant"() <{symbol = @c18446744073709551615, value = -1 : i64}> : () -> i64
    // CHECK: llh.extract
    // CHECK-SAME:-> tensor<?x?x?xf32, #llh.encoding<shapes = @s1, @s2, @s2>>
    %5 = "llh.extract"(%arg0, %3) : (tensor<?x?x?x?xf32>, i64) -> tensor<?x?x?xf32>
    // CHECK: llh.extract
    // CHECK-SAME:-> tensor<?x?xf32, #llh.encoding<shapes = @s2, @s2>>
    %6 = "llh.extract"(%5, %4) : (tensor<?x?x?xf32>, i64) -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
}

// -----
func.func @matmul(%arg0: tensor<?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1"}) -> tensor<*xf32> attributes {entrance} {
    // CHECK: llh.matmul
    // CHECK-SAME: tensor<?x10xf32, #llh.encoding<shapes = @s0, @c10>>
    %const = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<512x10xf32>
    %matmul = "llh.matmul"(%arg0, %const) : (tensor<?x?xf32>, tensor<512x10xf32>) -> tensor<*xf32>
    // llh.symbol_relation
    return %matmul : tensor<*xf32>
}

// -----
func.func @slice(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"} ) -> tensor<1x?x?x?xf32> attributes {entrance} {
    %c0_i64 = arith.constant {symbol = @c0} 0 : i64
    %c1_i64 = arith.constant {symbol = @c1} 1 : i64
    %c2_i64 = arith.constant {symbol = @c2} 2 : i64
    %c3_i64 = arith.constant {symbol = @c3} 3 : i64
    %0 = "llh.dim"(%arg0, %c1_i64) : (tensor<?x?x?x?xf32>, i64) -> i64
    %1 = "llh.dim"(%arg0, %c2_i64) : (tensor<?x?x?x?xf32>, i64) -> i64
    %2 = "llh.dim"(%arg0, %c3_i64) : (tensor<?x?x?x?xf32>, i64) -> i64
    %3 = "llh.slice"(%arg0, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %0, %1, %2, %c1_i64, %c1_i64, %c1_i64, %c1_i64) <{operandSegmentSizes = array<i32: 1, 4, 4, 4>}> : (tensor<?x?x?x?xf32>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> tensor<1x?x?x?xf32>
    return %3 : tensor<1x?x?x?xf32>
  }

// -----
// CHECK: "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
// CHECK: "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
// CHECK: "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
// CHECK: "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: block
// CHECK-SAME: (%arg0: tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, %arg1: bf16) -> tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>
func.func @block(%arg2: tensor<?x3x?x?xbf16>,%arg3: bf16) ->(tensor<?x3x?x?xbf16>) attributes {entrance}{
  return %arg2 : tensor<?x3x?x?xbf16>
}

