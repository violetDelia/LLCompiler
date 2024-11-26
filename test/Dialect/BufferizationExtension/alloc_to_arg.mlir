// RUN: llc-opt --split-input-file --alloc-to-arg -allow-unregistered-dialect %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --alloc-to-arg -allow-unregistered-dialect /home/lfr/LLCompiler/test/Dialect/BufferizationExtension/alloc_to_arg.mlir

func.func @main(%arg0: memref<?x?x?x?xf32> {bufferization.access = "read", func.input_symbol_0 = "s0", func.input_symbol_1 = "s1", func.input_symbol_2 = "s2", func.input_symbol_3 = "s2"}, %arg1: memref<?x?x?x?xf32> {bufferize.result}) attributes {entrance} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim_2 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim_3 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %dim_4 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
    %alloc = memref.alloc(%dim, %dim_2, %dim_3, %dim_4) {alignment = 64 : i64} : memref<?x?x?x?xf32>
    // CHECK-NOT: memref.copy
    memref.copy %alloc, %arg1 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }