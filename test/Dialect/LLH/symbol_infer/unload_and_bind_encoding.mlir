// RUN: llc-opt --split-input-file --unload-and-bind-encoding %s| FileCheck %s
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --unload-and-bind-encoding /home/lfr/LLCompiler/test/Dialect/LLH/unload_and_bind_encoding.mlir

"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: encoding
// CHECK-SAME: (%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT: "llh.encoding_bind"(%arg0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
// CHECK: llh.add
// CHECK-NEXT: "llh.encoding_bind"(%0) <{encoding = #llh.encoding<shapes = @s0, @s1, @s2, @s3>}> : (tensor<?x?x?x?xf32>) -> ()
func.func @encoding(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>> attributes {entrance} {
    %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
    return %0 : tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
}

// -----

"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK-LABEL: symbol
func.func @symbol(%arg0: tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> (i64) attributes {entrance} {
    %i0 = "llh.constant"() <{symbol = @c0,value = 0 : i64}> : () -> i64
    // CHECK: llh.symbol_bind
    // CHECK-SAME: symbol = @c0
    %104 = "llh.dim"(%arg0, %i0) <{symbol = @s0}> : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, i64) -> i64
    // CHECK: llh.symbol_bind
    // CHECK-SAME: symbol = @s0
    return %i0: i64
}




