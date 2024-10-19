// RUN: llc-opt --split-input-file --infer-symbol-shape --canonicalize %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --canonicalize /home/lfr/LLCompiler/test/Dialect/LLH/canonicalize_symbol.mlir 


"llh.symbolic_int"() <{sym_name = "s5"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s4"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c64"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
// CHECK: func.func
// CHECK-SAME: %arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
func.func @relation_eq(%arg0: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, %arg1: tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) ->(i64) attributes {entrance} {
  %2 = "llh.constant"() <{symbol = @c0, value = 2 : i64}> : () -> i64
  // CHECK: llh.add
  // CHECK-SAME: : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
  %0 = "llh.add"(%arg0, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s0, @c64, @s1, @s2>>
  %1 = "llh.add"(%arg1, %arg1) : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>
  "llh.encoding_bind"(%1) <{encoding = #llh.encoding<shapes = @s3, @c64, @s4, @s5>}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>) -> ()
  // CHECK: llh.dim
  // CHECK-SAME: <{symbol = @s1}>
  %193 = "llh.dim"(%1, %2) <{symbol = @s4}> : (tensor<?x64x?x?xf32, #llh.encoding<shapes = @s3, @c64, @s4, @s5>>, i64) -> i64
  return %193: i64
}
// CHECK-LABEL: __symbol__
// CHECK-NOT: llh.symbol_relation
module @__symbol__ {
  "llh.symbol_relation"() <{relation = @s5, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s2}> : () -> ()
  "llh.symbol_relation"() <{relation = @s4, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s1}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
  "llh.symbol_relation"() <{relation = @s3, relation_kind = #llh.SymbolRelation<EQ>, symbol = @s0}> : () -> ()
}

// -----

"llh.symbolic_int"() <{sym_name = "c2"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
// CHECK-LABEL: __remove_const__
module @__remove_const__ {
  // CHECK-NOT: llh.symbol_relation
  "llh.symbol_relation"() <{relation = @c1, relation_kind = #llh.SymbolRelation<GT>, symbol = @c2}> : () -> ()
  // CHECK-NOT: llh.symbol_binary_relation
  "llh.symbol_binary_relation"() <{relation_kind = #llh.SymbolRelation<Mul>, relations_lhs = @c512, relations_rhs = @c1, symbol = @c512}> : () -> ()
}





