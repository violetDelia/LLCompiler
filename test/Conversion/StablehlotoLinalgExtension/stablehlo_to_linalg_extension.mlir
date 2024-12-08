// RUN: llc-opt --split-input-file --convert-stablehlo-to-linalg-extension --canonicalize %s| FileCheck %s
//  /home/lfr/LLCompiler/build/bin/llc-opt --split-input-file --convert-stablehlo-to-linalg-extension --canonicalize /home/lfr/LLCompiler/test/Conversion/StablehlotoLinalgExtension/stablehlo_to_linalg_extension.mlir

func.func @main(%arg0: tensor<?x?x?x?xf32> {func.input_symbol_0 = "s0", func.input_symbol_1 = "s0", func.input_symbol_2 = "s1", func.input_symbol_3 = "s1"}) -> tensor<1x?x?x?xf32> attributes {entrance} {
    %cst = arith.constant dense<0> : tensor<4xindex>
    %cst_0 = arith.constant dense<1> : tensor<4xindex>
    %cst_1 = arith.constant dense<[1, 0, 0, 0]> : tensor<4xindex>
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim {symbol = @s0} %arg0, %c1 : tensor<?x?x?x?xf32>
    %dim_2 = tensor.dim {symbol = @s1} %arg0, %c2 : tensor<?x?x?x?xf32>
    %dim_3 = tensor.dim {symbol = @s1} %arg0, %c3 : tensor<?x?x?x?xf32>
    %from_elements = tensor.from_elements %c2, %dim, %dim_2, %dim_3 : tensor<4xindex>
    %0 = stablehlo.real_dynamic_slice %arg0, %cst_1, %from_elements, %cst_0 : (tensor<?x?x?x?xf32>, tensor<4xindex>, tensor<4xindex>, tensor<4xindex>) -> tensor<1x?x?x?xf32>
    return %0: tensor<1x?x?x?xf32>
}