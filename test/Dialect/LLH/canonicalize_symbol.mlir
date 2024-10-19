// RUN: llc-opt --split-input-file --infer-symbol-shape --canonicalize %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --canonicalize /home/lfr/LLCompiler/test/Dialect/LLH/canonicalize_symbol.mlir 


"llh.symbolic_int"() <{sym_name = "s5"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s4"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()

// CHECK-NOT: symbol = @s5
// CHECK-NOT: symbol = @s4
// CHECK-NOT: symbol = @s3
func.func @main(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) ->(i64) attributes {entrance} {
  %2 = "llh.constant"() <{symbol = @c0, value = 2 : i64}> : () -> i64
  %0 = "llh.add"(%arg0, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
  %1 = "llh.add"(%arg1, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>
  "llh.encoding_bind"(%1) <{encoding = #llh.encoding<shapes = @s3, @c64, @s4, @s5>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> ()
  %193 = "llh.dim"(%1, %2) <{symbol = @s4}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, i64) -> i64
  return %193: i64
}
module @__symbol__ {
  "llh.symbol_relation"() <{relation = @s5, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s2}> : () -> ()
  "llh.symbol_relation"() <{relation = @s4, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s1}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
}







