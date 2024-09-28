// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s


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
// CHECK: "llh.constant"
// CHECK-SAME: tensor<384xf32, #llh.encoding<shapes = @c384>>
// CHECK: "llh.constant"
// CHECK-SAME: -> i64
func.func @constant() ->() attributes {entrance}{
  %98 = "llh.constant"() <{value = dense<1.000000e+00> : tensor<384xf32>}> : () -> tensor<384xf32>
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
  %16 = "llh.reshape"(%arg0, %4, %3, %14, %15) : (tensor<?x?x224x226xf32, #llh.encoding<shapes = @s0, @s1, @c224, @c226>>, i64, i64, i64, i64) -> tensor<1x?x224x226xf32>
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  return 
}

