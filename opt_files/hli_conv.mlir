module {
  func.func @linalg.conv_0d_nc(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%1 : tensor<3x3xf32>) -> tensor<3x3xf32>
    return %2 : tensor<3x3xf32>
  }
}

// -----
module {
  func.func @linalg.conv_1d_nwc(%arg0: tensor<?x8x?xf32>, %arg1: tensor<2x?x?xf32>) -> tensor<?x7x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x8x?xf32>
    %dim_0 = tensor.dim %arg1, %c2 : tensor<2x?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x7x?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x7x?xf32>) -> tensor<?x7x?xf32>
    %2 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : tensor<1xi64>, someattr, strides = dense<1> : tensor<1xi64>} ins(%arg0, %arg1 : tensor<?x8x?xf32>, tensor<2x?x?xf32>) outs(%1 : tensor<?x7x?xf32>) -> tensor<?x7x?xf32>
    return %2 : tensor<?x7x?xf32>
  }
}

// -----
module {
  func.func @conv_2d_nhwc_hwcf(%arg0: tensor<?x4x5x?xf32>, %arg1: tensor<3x2x?x?xf32>) -> tensor<?x2x4x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x4x5x?xf32>
    %dim_0 = tensor.dim %arg1, %c3 : tensor<3x2x?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x2x4x?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x2x4x?xf32>) -> tensor<?x2x4x?xf32>
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<?x4x5x?xf32>, tensor<3x2x?x?xf32>) outs(%1 : tensor<?x2x4x?xf32>) -> tensor<?x2x4x?xf32>
    return %2 : tensor<?x2x4x?xf32>
  }
}

// -----
module {
  func.func @conv_transpose_2d(%arg0: tensor<2x9x10x3xf32>, %arg1: tensor<4x4x3x3xf32>) -> tensor<2x15x25x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x15x25x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x15x25x3xf32>) -> tensor<2x15x25x3xf32>
    %2 = tensor.empty() : tensor<2x21x31x3xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x21x31x3xf32>) -> tensor<2x21x31x3xf32>
    %inserted_slice = tensor.insert_slice %arg0 into %3[0, 6, 6, 0] [2, 9, 10, 3] [1, 1, 2, 1] : tensor<2x9x10x3xf32> into tensor<2x21x31x3xf32>
    %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%inserted_slice, %arg1 : tensor<2x21x31x3xf32>, tensor<4x4x3x3xf32>) outs(%1 : tensor<2x15x25x3xf32>) -> tensor<2x15x25x3xf32>
    return %4 : tensor<2x15x25x3xf32>
  }
}

// -----
module {
  func.func @conv_transpose_complex_2d(%arg0: tensor<2x9x10x3xcomplex<f32>>, %arg1: tensor<4x4x3x3xcomplex<f32>>) -> tensor<2x15x25x3xcomplex<f32>> {
    %cst = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
    %0 = tensor.empty() : tensor<2x15x25x3xcomplex<f32>>
    %1 = linalg.fill ins(%cst : complex<f32>) outs(%0 : tensor<2x15x25x3xcomplex<f32>>) -> tensor<2x15x25x3xcomplex<f32>>
    %2 = tensor.empty() : tensor<2x21x31x3xcomplex<f32>>
    %3 = linalg.fill ins(%cst : complex<f32>) outs(%2 : tensor<2x21x31x3xcomplex<f32>>) -> tensor<2x21x31x3xcomplex<f32>>
    %inserted_slice = tensor.insert_slice %arg0 into %3[0, 6, 6, 0] [2, 9, 10, 3] [1, 1, 2, 1] : tensor<2x9x10x3xcomplex<f32>> into tensor<2x21x31x3xcomplex<f32>>
    %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%inserted_slice, %arg1 : tensor<2x21x31x3xcomplex<f32>>, tensor<4x4x3x3xcomplex<f32>>) outs(%1 : tensor<2x15x25x3xcomplex<f32>>) -> tensor<2x15x25x3xcomplex<f32>>
    return %4 : tensor<2x15x25x3xcomplex<f32>>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2 + d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d2)>
module {
  func.func @conv_different_batch_dim_in_out(%arg0: tensor<1x1x1xf64>, %arg1: tensor<1x1x1xf64>) -> tensor<1x1x1xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty() : tensor<1x1x1xf64>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<1x1x1xf64>) -> tensor<1x1x1xf64>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : tensor<1x1x1xf64>, tensor<1x1x1xf64>) outs(%1 : tensor<1x1x1xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %3 = arith.mulf %in, %in_0 : f64
      %4 = arith.addf %out, %3 : f64
      linalg.yield %4 : f64
    } -> tensor<1x1x1xf64>
    return %2 : tensor<1x1x1xf64>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d3 + d4, d5 + d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d4, d6, d0, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d5, d7, d0, d2)>
module {
  func.func @conv_different_batch_dim_in_out_with_feature_group_count(%arg0: tensor<4x6x7x1xf64>, %arg1: tensor<2x6x3x2xf64>) -> tensor<1x2x1x2xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 0, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst : f64
    } : tensor<4x6x7x1xf64> to tensor<4x6x8x1xf64>
    %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 0] [4, 6, 6, 1] [1, 1, 1, 1] : tensor<4x6x8x1xf64> to tensor<4x6x6x1xf64>
    %0 = tensor.empty() : tensor<2x6x5x2xf64>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<2x6x5x2xf64>) -> tensor<2x6x5x2xf64>
    %inserted_slice = tensor.insert_slice %arg1 into %1[0, 0, 0, 0] [2, 6, 3, 2] [1, 1, 2, 1] : tensor<2x6x3x2xf64> into tensor<2x6x5x2xf64>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1], [2], [3], [4]] output_shape [2, 2, 6, 6, 1] : tensor<4x6x6x1xf64> into tensor<2x2x6x6x1xf64>
    %expanded_0 = tensor.expand_shape %inserted_slice [[0], [1], [2], [3, 4]] output_shape [2, 6, 5, 2, 1] : tensor<2x6x5x2xf64> into tensor<2x6x5x2x1xf64>
    %2 = tensor.empty() : tensor<1x2x1x2x1xf64>
    %3 = linalg.fill ins(%cst : f64) outs(%2 : tensor<1x2x1x2x1xf64>) -> tensor<1x2x1x2x1xf64>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%expanded, %expanded_0 : tensor<2x2x6x6x1xf64>, tensor<2x6x5x2x1xf64>) outs(%3 : tensor<1x2x1x2x1xf64>) {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %5 = arith.mulf %in, %in_1 : f64
      %6 = arith.addf %out, %5 : f64
      linalg.yield %6 : f64
    } -> tensor<1x2x1x2x1xf64>
    %collapsed = tensor.collapse_shape %4 [[0], [1], [2], [3, 4]] : tensor<1x2x1x2x1xf64> into tensor<1x2x1x2xf64>
    return %collapsed : tensor<1x2x1x2xf64>
  }
}

// -----
module {
  func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x8x8x8x?xf32>, %arg1: tensor<2x2x2x?x?xf32>) -> tensor<?x7x7x7x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x8x8x8x?xf32>
    %dim_0 = tensor.dim %arg1, %c4 : tensor<2x2x2x?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x7x7x7x?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x7x7x7x?xf32>) -> tensor<?x7x7x7x?xf32>
    %2 = linalg.conv_3d_ndhwc_dhwcf {dilations = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} ins(%arg0, %arg1 : tensor<?x8x8x8x?xf32>, tensor<2x2x2x?x?xf32>) outs(%1 : tensor<?x7x7x7x?xf32>) -> tensor<?x7x7x7x?xf32>
    return %2 : tensor<?x7x7x7x?xf32>
  }
}

// -----
module {
  func.func @conv2d_1452x2223_dilated_valid(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>) -> tensor<1x2x4x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x2x4x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>) outs(%1 : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
    return %2 : tensor<1x2x4x3xf32>
  }
}

// -----
module {
  func.func @linalg.conv_2D_padding_test1(%arg0: tensor<1x33x1x1xf16>, %arg1: tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<400x1024x1024x1xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
    %padded = tensor.pad %arg1 low[0, 0, 16, 0] high[0, 0, 16, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst : f16
    } : tensor<400x1024x1024x1xf16> to tensor<400x1024x1056x1xf16>
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %arg0 : tensor<400x1024x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%1 : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
    return %2 : tensor<400x1024x1024x1xf16>
  }
}

// -----
module {
  func.func @linalg.conv_2D_padding_test2(%arg0: tensor<1x33x1x1xf16>, %arg1: tensor<400x1024x1024x1xf16>) -> tensor<400x1040x1024x1xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<400x1040x1024x1xf16>
    %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
    %padded = tensor.pad %arg1 low[0, 8, 16, 0] high[0, 8, 16, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst : f16
    } : tensor<400x1024x1024x1xf16> to tensor<400x1040x1056x1xf16>
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %arg0 : tensor<400x1040x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%1 : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
    return %2 : tensor<400x1040x1024x1xf16>
  }
}

// -----
module {
  func.func @depthwise_conv(%arg0: tensor<2x4x5x2xf32>, %arg1: tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<2x2x1x6xf32> into tensor<24xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1, 2, 3]] output_shape [2, 2, 2, 3] : tensor<24xf32> into tensor<2x2x2x3xf32>
    %0 = tensor.empty() : tensor<2x3x4x2x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, someattr, strides = dense<1> : tensor<2xi64>} ins(%arg0, %expanded : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>) outs(%1 : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
    %collapsed_0 = tensor.collapse_shape %2 [[0], [1], [2], [3, 4]] : tensor<2x3x4x2x3xf32> into tensor<2x3x4x6xf32>
    return %collapsed_0 : tensor<2x3x4x6xf32>
  }
}

// -----
module {
  func.func @depthwise_conv_with_padding(%arg0: tensor<2x4x5x2xf32>, %arg1: tensor<2x2x1x4xf32>) -> tensor<2x3x6x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 0] high[0, 0, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst : f32
    } : tensor<2x4x5x2xf32> to tensor<2x4x7x2xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2, 3]] : tensor<2x2x1x4xf32> into tensor<16xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1, 2, 3]] output_shape [2, 2, 2, 2] : tensor<16xf32> into tensor<2x2x2x2xf32>
    %0 = tensor.empty() : tensor<2x3x6x2x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, someattr, strides = dense<1> : tensor<2xi64>} ins(%padded, %expanded : tensor<2x4x7x2xf32>, tensor<2x2x2x2xf32>) outs(%1 : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
    %collapsed_0 = tensor.collapse_shape %2 [[0], [1], [2], [3, 4]] : tensor<2x3x6x2x2xf32> into tensor<2x3x6x4xf32>
    return %collapsed_0 : tensor<2x3x6x4xf32>
  }
}

// -----
module {
  func.func @depthwise_conv_multiplier_1(%arg0: tensor<1x113x113x96xf32>, %arg1: tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x56x56x96xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %collapsed : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>) outs(%1 : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
    return %2 : tensor<1x56x56x96xf32>
  }
}

// -----
module {
  func.func @depthwise_conv_multiplier_1_with_padding(%arg0: tensor<1x113x113x96xf32>, %arg1: tensor<3x3x1x96xf32>) -> tensor<1x57x58x96xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 1, 2, 0] high[0, 1, 2, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst : f32
    } : tensor<1x113x113x96xf32> to tensor<1x115x117x96xf32>
    %0 = tensor.empty() : tensor<1x57x58x96xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2, 3]] : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
    %2 = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%padded, %collapsed : tensor<1x115x117x96xf32>, tensor<3x3x96xf32>) outs(%1 : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>
    return %2 : tensor<1x57x58x96xf32>
  }
}

// -----
module {
  func.func @depthwise_conv1d(%arg0: tensor<1x10x8xf32>, %arg1: tensor<3x1x16xf32>) -> tensor<1x10x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 1, 0] high[0, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x10x8xf32> to tensor<1x12x8xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2]] : tensor<3x1x16xf32> into tensor<48xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1, 2]] output_shape [3, 8, 2] : tensor<48xf32> into tensor<3x8x2xf32>
    %0 = tensor.empty() : tensor<1x10x8x2xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x10x8x2xf32>) -> tensor<1x10x8x2xf32>
    %2 = linalg.depthwise_conv_1d_nwc_wcm {dilations = dense<1> : tensor<1xi64>, someattr, strides = dense<1> : tensor<1xi64>} ins(%padded, %expanded : tensor<1x12x8xf32>, tensor<3x8x2xf32>) outs(%1 : tensor<1x10x8x2xf32>) -> tensor<1x10x8x2xf32>
    %collapsed_0 = tensor.collapse_shape %2 [[0], [1], [2, 3]] : tensor<1x10x8x2xf32> into tensor<1x10x16xf32>
    return %collapsed_0 : tensor<1x10x16xf32>
  }
}

// -----
module {
  func.func @depthwise_conv1d_m1(%arg0: tensor<1x10x8xf32>, %arg1: tensor<3x1x8xf32>) -> tensor<1x10x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 1, 0] high[0, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x10x8xf32> to tensor<1x12x8xf32>
    %0 = tensor.empty() : tensor<1x10x8xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<3x1x8xf32> into tensor<3x8xf32>
    %2 = linalg.depthwise_conv_1d_nwc_wc {dilations = dense<1> : tensor<1xi64>, someattr, strides = dense<1> : tensor<1xi64>} ins(%padded, %collapsed : tensor<1x12x8xf32>, tensor<3x8xf32>) outs(%1 : tensor<1x10x8xf32>) -> tensor<1x10x8xf32>
    return %2 : tensor<1x10x8xf32>
  }
}

// -----
module {
  func.func @depthwise_conv3d(%arg0: tensor<2x3x5x4x6xf32>, %arg1: tensor<2x1x3x1x36xf32>) -> tensor<2x3x13x4x36xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 1, 5, 3, 0] high[0, 2, 3, 5, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<2x3x5x4x6xf32> to tensor<2x6x13x12x6xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2, 3, 4]] : tensor<2x1x3x1x36xf32> into tensor<216xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1, 2, 3, 4]] output_shape [2, 1, 3, 6, 6] : tensor<216xf32> into tensor<2x1x3x6x6xf32>
    %0 = tensor.empty() : tensor<2x3x13x4x6x6xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
    %2 = linalg.depthwise_conv_3d_ndhwc_dhwcm {dilations = dense<1> : tensor<3xi64>, someattr, strides = dense<[2, 1, 3]> : tensor<3xi64>} ins(%padded, %expanded : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6x6xf32>) outs(%1 : tensor<2x3x13x4x6x6xf32>) -> tensor<2x3x13x4x6x6xf32>
    %collapsed_0 = tensor.collapse_shape %2 [[0], [1], [2], [3], [4, 5]] : tensor<2x3x13x4x6x6xf32> into tensor<2x3x13x4x36xf32>
    return %collapsed_0 : tensor<2x3x13x4x36xf32>
  }
}

// -----
module {
  func.func @depthwise_conv3d_m1(%arg0: tensor<2x3x5x4x6xf32>, %arg1: tensor<2x1x3x1x6xf32>) -> tensor<2x3x13x4x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 1, 5, 3, 0] high[0, 2, 3, 5, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<2x3x5x4x6xf32> to tensor<2x6x13x12x6xf32>
    %0 = tensor.empty() : tensor<2x3x13x4x6xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
    %collapsed = tensor.collapse_shape %arg1 [[0], [1], [2], [3, 4]] : tensor<2x1x3x1x6xf32> into tensor<2x1x3x6xf32>
    %2 = linalg.depthwise_conv_3d_ndhwc_dhwc {dilations = dense<1> : tensor<3xi64>, someattr, strides = dense<[2, 1, 3]> : tensor<3xi64>} ins(%padded, %collapsed : tensor<2x6x13x12x6xf32>, tensor<2x1x3x6xf32>) outs(%1 : tensor<2x3x13x4x6xf32>) -> tensor<2x3x13x4x6xf32>
    return %2 : tensor<2x3x13x4x6xf32>
  }
}

