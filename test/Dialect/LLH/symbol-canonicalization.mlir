// RUN: llc-opt --split-input-file --symbol-canonicalization %s| FileCheck %s

module attributes {}{
  // CHECK: "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
  // CHECK-LABEL: replaceTorchSymbolicIntOp
  func.func @replaceTorchSymbolicIntOp() ->(i64) {
  // CHECK: %[[VAR0:.*]] = "llh.torch_symbolic_int"() <{sym_name = "s0"}> {symbol_generated} : () -> i64
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
  return  %123: i64
  }
}


// -----

#map8 = affine_map<()[s0, s1] -> (s0, 1000)>
module attributes {}{
  // CHECK: "llh.symbolic_int"() <{sym_name = "c1000"}> : () -> ()
  // CHECK-LABEL: replaceTorchSymbolicIntOp
  func.func @replaceTorchSymbolicIntOp(%arg0: tensor<1000xf32>, %arg1: tensor<?x1000xf32>) ->() {
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s12"}> : () -> i64
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s19"}> : () -> i64
  %195 = "llh.add"(%arg1, %arg0) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
  "llh.symbolic_bind"(%195, %123) <{expressions = #map8}> {} : (tensor<?x1000xf32>, i64) -> ()
  return 
  }
}


