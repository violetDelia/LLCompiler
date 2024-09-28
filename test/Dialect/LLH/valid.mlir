// RUN: llc-opt %s | llc-opt | FileCheck %s

// -----
// CHECK-LABEL: torch_symbolic_int
func.func @torch_symbolic_int() -> (i64) {
  %6 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
  return %6 : i64
}