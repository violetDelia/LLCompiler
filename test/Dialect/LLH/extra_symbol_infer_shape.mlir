// RUN: llc-opt --split-input-file --infer-symbol-shape %s| FileCheck %s


"llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c512"}> : () -> ()
"llh.symbolic_int"() <{sym_name = "c1"}> : () -> ()
// CHECK-LABEL: dim_and_const
func.func @dim_and_const(%arg0: tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>) -> () attributes {entrance} {
  %0 = "llh.constant"() <{value = 3 : i64}> : () -> i64
  %1 = "llh.constant"() <{value = 2 : i64}> : () -> i64
  %2 = "llh.constant"() <{value = 0 : i64}> : () -> i64
  %3 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c3}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c2}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c0}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c1}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @s0}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c512}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c1}>
  // CHECK: llh.symbol_bind
  // CHECK-SAME: <{symbol = @c1}>
  %193 = "llh.dim"(%arg0, %2) : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %194 = "llh.dim"(%arg0, %3) : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %195 = "llh.dim"(%arg0, %1) : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  %196 = "llh.dim"(%arg0, %0) : (tensor<?x512x1x1xf32, #llh.encoding<shapes = @s0, @c512, @c1, @c1>>, i64) -> i64
  return 
}
