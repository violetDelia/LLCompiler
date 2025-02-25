// RUN: llc-opt %s -one-shot-bufferize="bufferize-function-boundaries"  -canonicalize -buffer-loop-hoisting -drop-equivalent-buffer-results -split-input-file | FileCheck %s

func.func @print(){
    %const = arith.constant dense<1.0> : tensor<2xf32>
    // CHECK: llh.print
    // CHECK-SAME: memref<2xf32>
    "llh.print"(%const) <{prefix_description = "print const: "}>: (tensor<2xf32>) -> ()
    return 
}

//  /home/lfr/LLCompiler/build/bin/llc-opt /home/lfr/LLCompiler/test/Dialect/LLH/one-shot-bufferize.mlir -one-shot-bufferize="bufferize-function-boundaries" -canonicalize -buffer-loop-hoisting -drop-equivalent-buffer-results -split-input-file
