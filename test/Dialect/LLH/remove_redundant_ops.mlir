// RUN: llc-opt --split-input-file --remove-redundant-ops %s| FileCheck %s


func.func @replaceFlattenOp(%arg2: tensor<?x512x1x1xf32>) ->() {
  // CHECK: %[[Reshape:.*]] = "llh.reshape"(%[[VAR0:.*]], %[[ND1:.*]], %[[ND2:.*]]) : (tensor<?x512x1x1xf32>, i64, i64) -> tensor<?x512xf32>
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %192 = "llh.flatten"(%arg2, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
  return 
}

// -----
module attributes {builtin.gloabal_layout = "NCHW"}{
  // CHECK-LABEL: replaceTorchSymbolicIntOp
  func.func @replaceTorchSymbolicIntOp() ->() {
  // CHECK-NOT: llh.torch_symbolic_int
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
  return 
  }
}

// -----
#map8 = affine_map<()[s0, s1] -> (s0, 1000)>
module attributes {builtin.gloabal_layout = "NCHW"}{
  // CHECK-LABEL: replaceTorchSymbolicIntOp
  func.func @replaceTorchSymbolicIntOp(%arg0: tensor<1000xf32>, %arg1: tensor<?x1000xf32>) ->() {
  // CHECK-NOT: llh.torch_symbolic_int
  // CHECK-NOT: llh.symbolic_bind
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s12"}> : () -> i64
  %124 = "llh.torch_symbolic_int"() <{sym_name = "s19"}> : () -> i64
  %195 = "llh.add"(%arg1, %arg0) : (tensor<?x1000xf32>, tensor<1000xf32>) -> tensor<?x1000xf32>
  "llh.symbolic_bind"(%195, %123) <{expressions = #map8}> {} : (tensor<?x1000xf32>, i64) -> ()
  return 
  }
}
