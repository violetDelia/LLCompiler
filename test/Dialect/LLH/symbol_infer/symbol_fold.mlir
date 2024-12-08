// RUN: llc-opt --split-input-file -infer-symbol-shape="use-encoding=false" --symbol-fold %s| FileCheck %s
// /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file -infer-symbol-shape="use-encoding=false" --symbol-fold /home/lfr/LLCompiler/test/Dialect/LLH/symbol_infer/symbol_fold.mlir



// CHECK-LABEL: tensor_dim_fold
func.func @tensor_dim_fold(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> index attributes {entrance} {
    %c3 = arith.constant 3 : index
    // CHECK-COUNT-1: tensor.dim
    %dim_1 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
    %dim_5 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
    %6 = arith.addi %dim_5, %dim_1 : index
    return %6: index
  }

// -----
// CHECK-LABEL: arith_to_const
func.func @arith_to_const(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> index attributes {entrance} {
    %c3 = arith.constant 3 : index
    %dim_1 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
    %dim_5 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
    // CHECK: arith.constant 0 : index
    %6 = arith.subi %dim_5, %dim_1 : index
    return %6: index
  }