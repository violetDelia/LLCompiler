// -----// IR Dump After FinalizingBufferize (finalizing-bufferize) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x10xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2], [3]] : memref<1x1x28x28xf32> into memref<1x28x28xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 28, 28, 1] : memref<1x28x28xf32> into memref<1x28x28x1xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc_1 : memref<1x32x32x1xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  }
  %subview = memref.subview %alloc_1[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc_2 : memref<1x28x28x8xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%alloc_1, %3 : memref<1x32x32x1xf32>, memref<8x5x5x1xf32>) outs(%alloc_2 : memref<1x28x28x8xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %6 = arith.mulf %in, %in_16 : f32
    %7 = arith.addf %out, %6 : f32
    linalg.yield %7 : f32
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc_3 : memref<1x14x14x8xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst_0 : f32
  }
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d1 * 2 + d4, d2 * 2 + d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d3, 0, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%alloc_2, %2, %alloc_4 : memref<1x28x28x8xf32>, memref<1x8x1x1xf32>, memref<2x2xf32>) outs(%alloc_3 : memref<1x14x14x8xf32>) {
  ^bb0(%in: f32, %in_16: f32, %in_17: f32, %out: f32):
    %6 = arith.addf %in, %in_16 : f32
    %7 = arith.maximumf %6, %cst : f32
    %8 = arith.maximumf %out, %7 : f32
    linalg.yield %8 : f32
  }
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc_5 : memref<1x18x18x8xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  }
  %subview_6 = memref.subview %alloc_5[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_3, %subview_6 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc_7 : memref<1x14x14x16xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%alloc_5, %0 : memref<1x18x18x8xf32>, memref<16x5x5x8xf32>) outs(%alloc_7 : memref<1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %6 = arith.mulf %in, %in_16 : f32
    %7 = arith.addf %out, %6 : f32
    linalg.yield %7 : f32
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%alloc_8 : memref<1x4x4x16xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst_0 : f32
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<3x3xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d1 * 3 + d4, d2 * 3 + d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d3, 0, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%alloc_7, %1, %alloc_9 : memref<1x14x14x16xf32>, memref<1x16x1x1xf32>, memref<3x3xf32>) outs(%alloc_8 : memref<1x4x4x16xf32>) {
  ^bb0(%in: f32, %in_16: f32, %in_17: f32, %out: f32):
    %6 = arith.addf %in, %in_16 : f32
    %7 = arith.maximumf %6, %cst : f32
    %8 = arith.maximumf %out, %7 : f32
    linalg.yield %8 : f32
  }
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_8 : memref<1x4x4x16xf32>) outs(%alloc_10 : memref<1x16x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  %collapse_shape_11 = memref.collapse_shape %alloc_10 [[0], [1, 2, 3]] : memref<1x16x4x4xf32> into memref<1x256xf32>
  %expand_shape_12 = memref.expand_shape %collapse_shape_11 [[0, 1], [2]] output_shape [1, 1, 256] : memref<1x256xf32> into memref<1x1x256xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_13 : memref<1x1x10xf32>) {
  ^bb0(%out: f32):
    linalg.yield %cst : f32
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expand_shape_12, %4 : memref<1x1x256xf32>, memref<1x256x10xf32>) outs(%alloc_13 : memref<1x1x10xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %6 = arith.mulf %in, %in_16 : f32
    %7 = arith.addf %out, %6 : f32
    linalg.yield %7 : f32
  }
  %collapse_shape_14 = memref.collapse_shape %alloc_13 [[0, 1], [2]] : memref<1x1x10xf32> into memref<1x10xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%collapse_shape_14, %5 : memref<1x10xf32>, memref<1x10xf32>) outs(%alloc_15 : memref<1x10xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %6 = arith.addf %in, %in_16 : f32
    linalg.yield %6 : f32
  }
  memref.copy %alloc_15, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  return
}

