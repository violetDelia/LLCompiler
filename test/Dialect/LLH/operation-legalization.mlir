// RUN: llc-opt --operation-legalization %s | FileCheck %s



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


