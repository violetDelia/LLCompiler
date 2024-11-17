// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --infer-symbol-shape /home/lfr/LLCompiler/test/Dialect/LLH/extra_symbol_infer_shape.mlir

"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
// CHECK-LABEL: const
func.func @const(%arg0: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c3
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c2
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c0
  // CHECK: llh.constant
  // CHECK-SAME: symbol = @c1
  return 
}

// -----

// CHECK-LABEL: mul
func.func @mul(%arg0: tensor<?x512x1x1xf32>) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %193 = "llh.dim"(%arg0, %2) : (tensor<?x512x1x1xf32>, i64) -> i64
  %194 = "llh.dim"(%arg0, %3) : (tensor<?x512x1x1xf32>, i64) -> i64
  %195 = "llh.dim"(%arg0, %1) : (tensor<?x512x1x1xf32>, i64) -> i64
  %196 = "llh.dim"(%arg0, %0) : (tensor<?x512x1x1xf32>, i64) -> i64
  // CHECK: llh.mul
  // CHECK-SAME: symbol = @s1
  // CHECK: llh.mul
  // CHECK-SAME: symbol = @c512
  // CHECK: llh.mul
  // CHECK-SAME: symbol = @c1
  %197 = "llh.mul"(%193, %195) : (i64, i64) -> i64
  %198 = "llh.mul"(%194, %195) : (i64, i64) -> i64
  %199 = "llh.mul"(%196, %195) : (i64, i64) -> i64
  return 
}
// CHECK-LABEL: __symbol__
// CHECK: llh.symbol_binary_relation
// CHECK-SAME: relation_kind = #llh.SymbolRelation<Mul>