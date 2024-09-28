// RUN: llc-opt %s | llc-opt | FileCheck %s
// RUN: llc-opt %s --mlir-print-op-generic | llc-opt | FileCheck %s

// CHECK-LABEL: llh.symbolic_int
"llh.symbolic_int"() <{sym_name = "c1000"}> : () -> () 

// CHECK-LABEL: torch_symbolic_int
func.func @torch_symbolic_int() -> () {
    %6 = "llh.torch_symbolic_int"() <{sym_name = "s0"}> : () -> i64
    return
}

// CHECK-LABEL: aot
func.func @aot(%arg0: i64, %arg1: i64, %arg2: tensor<?x?x224x224xf32>) ->(){
    %1 = "llh.aot"(%arg0) <{name = "aot1"}> : (i64) -> i64
    %2 = "llh.aot"(%arg1,%arg2) <{name = "aot2"}> : (i64,tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
    %3 = "llh.aot"() <{name = "aot3"}> : () -> i64
    %4 = "llh.aot"() <{name = "aot4"}> : () -> i64
    %5 = "llh.aot"() <{name = "aot5"}> : () -> i64
    return
}

