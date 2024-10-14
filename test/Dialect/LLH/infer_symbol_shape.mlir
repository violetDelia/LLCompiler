// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --infer-symbol-shape /home/lfr/LLCompiler/test/Dialect/LLH/infer_symbol_shape.mlir

// CHECK: "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
// CHECK: "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
// CHECK: "llh.symbolic_int"() <{sym_name = "c3"}> : () -> ()
// CHECK: "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: block
// CHECK-SAME: (%arg0: tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, %arg1: bf16) -> tensor<?x3x?x?xbf16, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>
func.func @block(%arg2: tensor<?x3x?x?xbf16>,%arg3: bf16) ->(tensor<?x3x?x?xbf16>) attributes {entrance}{
  return %arg2 : tensor<?x3x?x?xbf16>
}

// -----
// CHECK: "llh.symbolic_int"() <{sym_name = "c384"}> : () -> ()
// CHECK-LABEL: constant
func.func @constant() ->() attributes {entrance}{
  // CHECK: llh.constant
  // CHECK-SAME: tensor<384xf32, #llh.encoding<shapes = @c384>>
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
  // CHECK: llh.constant
  // CHECK-SAME: -> i64
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
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
  %16 = "llh.reshape"(%arg0, %4, %3, %14, %15) : (tensor<?x?x224x226xf32, #llh.encoding<shapes = @s0, @s1, @c224, @c226>>, i64, i64, i64, i64) -> tensor<1x?x224x226xf32>
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
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
  %126 = "llh.conv"(%arg0, %4) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 7, 7>, layout = #llh.Layout<NCHW>, pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>}> : (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, tensor<64x3x7x7xf32, #llh.encoding<shapes = @c64, @c3, @c7, @c7>>) -> tensor<?x64x?x?xf32>
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
func.func @binary(%arg0: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, %arg2: tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s3, @s1, @s3>>) ->() attributes {entrance}{
  %4 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
  // CHECK: llh.add
  // CHECK-SAME: tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>> 
  %126 = "llh.add"(%arg0, %4): (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, tensor<1xf32, #llh.encoding<shapes = @c1>>) -> tensor<?x3x?x?xf32>
  // CHECK: llh.sub
  // CHECK-SAME: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s4, @s1, @s5>>
  %127 = "llh.sub"(%arg0, %arg2): (tensor<?x3x?x?xf32, #llh.encoding<shapes = @s0, @c3, @s1, @s2>>, tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s3, @s1, @s3>>) -> tensor<?x?x?x?xf32>

  // CHECK: llh.add
  // CHECK-SAME: tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>
  %6 = "llh.weight"() <{weight_file = "xxx"}> : () -> tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>
  %125 = "llh.add"(%6, %6): (tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>, tensor<1x3xf32, #llh.encoding<shapes = @c1, @c3>>) -> tensor<1x3xf32>
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
func.func @empty(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> () attributes {entrance} {
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
  return 
}

// -----
"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: max_pool
func.func @max_pool(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>) -> () attributes {entrance} {
 // CHECK: llh.max_pool
 // CHECK-SAME: -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @s0, @s2, @s4>>
 %129 = "llh.max_pool"(%arg0) <{ceil_mode = false, dilation = array<i64: 1, 1>, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s3>>) -> tensor<?x64x?x?xf32>
  return 
}

