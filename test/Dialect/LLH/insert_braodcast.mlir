// RUN: llc-opt --split-input-file --canonicalize %s| FileCheck %s

"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
func.func @main(%arg0: tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>) -> tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>> attributes {entrance} {
    %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "llh.constant"() <{value = dense<2.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32, #llh.encoding<shapes = @c1>>
    %15 = "llh.reshape"(%2, %0, %0, %0, %0) : (tensor<1xf32, #llh.encoding<shapes = @c1>>, i64, i64, i64, i64) -> tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>
    // CHECK:  llh.broadcast_to
    // CHECK-SAME: <{cast_dims = array<i64: 0, 1, 2, 3>, expand_dims = array<i64: 1, 2, 3>, noexpand_dims = array<i64: 0>}> : (tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>, i64, i64, i64, i64) -> tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>
    // CHECK:  llh.div
    // CHECK-SAME: (tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>, tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>) -> tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>
    %16 = "llh.div"(%arg0, %15) : (tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>, tensor<1x1x1x1xf32, #llh.encoding<shapes = @c1, @c1, @c1, @c1>>) -> tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>
    return  %16  : tensor<1x?x?x?xf32, #llh.encoding<shapes = @c1, @s1, @s2, @s3>>
  }

// -----
// CHECK-LABEL: where
func.func @where(%arg0: tensor<1x?x?x?xi1>) -> tensor<1x?x?x?xi64> attributes {entrance} {
    %1 = "llh.constant"() <{value = dense<-1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = "llh.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3 = "llh.where"(%arg0, %0, %1) : (tensor<1x?x?x?xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?x?xi64>
    return  %3  : tensor<1x?x?x?xi64>
  }

// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --insert-broadcast /home/lfr/LLCompiler/test/Dialect/LLH/insert_braodcast.mlir