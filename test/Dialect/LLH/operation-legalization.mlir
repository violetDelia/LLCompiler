// RUN: llc-opt --operation-legalization %s | FileCheck %s

func.func @replaceFlattenOp(%arg2: tensor<?x512x1x1xf32>) ->() {
  // CHECK: %[[Reshape:.*]] = "llh.reshape"(%[[VAR0:.*]], %[[ND1:.*]], %[[ND2:.*]]) : (tensor<?x512x1x1xf32>, i64, i64) -> tensor<?x512xf32>
  %0 = "llh.constant"() <{value = 1 : i64}> : () -> i64
  %192 = "llh.flatten"(%arg2, %0) : (tensor<?x512x1x1xf32>, i64) -> tensor<?x512xf32>
  return 
}

func.func @BraodcastableScalarToTensor1(%arg2: tensor<?x3x?x?xi64>) ->() {
  // CHECK: %[[VAR0:.*]] = "llh.constant"() <{value = dense<1> : tensor<1xi64>}>
  // CHECK-SAME: -> tensor<1xi64>
  // CHECK: %[[VAR2:.*]] = "llh.add"(%[[VAR3:.*]], %[[VAR0]])
  // CHECK-SAME: (tensor<?x3x?x?xi64>, tensor<1xi64>) -> tensor<?x3x?x?xi64>
  %0 = "llh.constant"() <{value = 1.0 : f32}> : () -> f32
  %163 = "llh.add"(%arg2, %0) : (tensor<?x3x?x?xi64>, f32) -> tensor<?x3x?x?xi64>
  return 
}

func.func @BraodcastableScalarToTensor2(%arg2: tensor<?x3x?x?xbf16>) ->() {
  // CHECK: %[[VAR0:.*]] = "llh.constant"() <{value = dense<1.000000e+00> : tensor<1xbf16>}>
  // CHECK-SAME: -> tensor<1xbf16>
  // CHECK: %[[VAR2:.*]] = "llh.add"(%[[VAR3:.*]], %[[VAR0]])
  // CHECK-SAME: (tensor<?x3x?x?xbf16>, tensor<1xbf16>) -> tensor<?x3x?x?xbf16>
  %0 = "llh.constant"() <{value = 1.0 : f32}> : () -> f32
  %163 = "llh.add"(%arg2, %0) : (tensor<?x3x?x?xbf16>, f32) -> tensor<?x3x?x?xbf16>
  return 
}

module attributes {}{
  // CHECK-LABEL: replaceTorchSymbolicIntOp
  func.func @replaceTorchSymbolicIntOp() ->() {
  // CHECK-NOT: llh.torch_symbolic_int
  %123 = "llh.torch_symbolic_int"() <{sym_name = "s2"}> : () -> i64
  return 
  }
}


// -----

#map8 = affine_map<()[s0, s1] -> (s0, 1000)>
module attributes {}{
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


