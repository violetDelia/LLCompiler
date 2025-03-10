// RUN: llc-opt %s -transform-pipeline | mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils | FileCheck %s

module{
    func.func @main() -> () attributes {entrance} {
        %const = "llh.constant"() <{value = dense<-1.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
        %0 = "llh.abs"(%const) : (tensor<2xf32>) -> tensor<2xf32>
        // CHECK: abs
        // CHECK: Unranked Memref
        // CHECK-SAME: rank = 1 offset = 0 sizes = [2] strides = [1] data = 
        // CHECK-NEXT: [1,  1]
        "llh.print"(%0) <{prefix_description = "abs"}>: (tensor<2xf32>) -> ()
        return 
  }
}