// -----// IR Dump After SimplifyAffineStructures (affine-simplify-structures) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x10xf32>) {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %alloc_3 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_4 = arith.constant 0 : index
  %c0_5 = arith.constant 0 : index
  %alloc_6 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_7 = arith.constant 0 : index
  %c0_8 = arith.constant 0 : index
  %c0_9 = arith.constant 0 : index
  %c0_10 = arith.constant 0 : index
  %c0_11 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_12 = arith.constant -3.40282347E+38 : f32
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc_13 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %collapse_shape = memref.collapse_shape %alloc_13 [[0, 1], [2], [3]] : memref<1x1x28x28xf32> into memref<1x28x28xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 28, 28, 1] : memref<1x28x28xf32> into memref<1x28x28x1xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %c1024 = arith.constant 1024 : index
  affine.parallel (%arg2) = (0) to (symbol(%c1024)) {
    %c0_25 = arith.constant 0 : index
    %6 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 32)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) floordiv 32)>(%arg2)
    affine.store %cst, %alloc_14[%8, %7, %6, %c0_25] : memref<1x32x32x1xf32>
  }
  %subview = memref.subview %alloc_14[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg2 = 0 to 1568 {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 14)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 14)>(%arg2)
    %c0_25 = arith.constant 0 : index
    affine.store %cst_12, %alloc_15[%c0_10, %8, %7, %6] : memref<1x14x14x8xf32>
    affine.for %arg3 = 0 to 4 {
      affine.store %cst, %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.for %arg4 = 0 to 25 {
        %15 = affine.apply affine_map<(d0) -> (d0 mod 5)>(%arg4)
        %16 = affine.apply affine_map<(d0) -> (d0 floordiv 5)>(%arg4)
        %17 = affine.apply affine_map<(d0, d1, d2) -> (((d2 floordiv 8) floordiv 14) * 2 + d1 floordiv 2 + d0 floordiv 5)>(%arg4, %arg3, %arg2)
        %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + (d2 floordiv 8) * 2 - ((d2 floordiv 8) floordiv 14) * 28 - (d1 floordiv 2) * 2 - (d0 floordiv 5) * 5)>(%arg4, %arg3, %arg2)
        %19 = affine.load %alloc_14[%c0_9, %17, %18, %c0_8] : memref<1x32x32x1xf32>
        %20 = affine.load %3[%6, %16, %15, %c0_8] : memref<8x5x5x1xf32>
        %21 = affine.load %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %22 = arith.mulf %19, %20 : f32
        %23 = arith.addf %21, %22 : f32
        affine.store %23, %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %9 = affine.load %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %10 = affine.load %2[%c0_11, %6, %c0_11, %c0_11] : memref<1x8x1x1xf32>
      %11 = affine.load %alloc_15[%c0_25, %8, %7, %6] : memref<1x14x14x8xf32>
      %12 = arith.addf %9, %10 : f32
      %13 = arith.maximumf %12, %cst : f32
      %14 = arith.maximumf %11, %13 : f32
      affine.store %14, %alloc_15[%c0_25, %8, %7, %6] : memref<1x14x14x8xf32>
    }
  }
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %c2592 = arith.constant 2592 : index
  affine.parallel (%arg2) = (0) to (symbol(%c2592)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 18)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 18) mod 18)>(%arg2)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 18) floordiv 18)>(%arg2)
    affine.store %cst, %alloc_16[%9, %8, %7, %6] : memref<1x18x18x8xf32>
  }
  %subview_17 = memref.subview %alloc_16[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_15, %subview_17 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg2 = 0 to 256 {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) mod 4)>(%arg2)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) floordiv 4)>(%arg2)
    %c0_25 = arith.constant 0 : index
    affine.store %cst_12, %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.for %arg3 = 0 to 9 {
      affine.store %cst, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.for %arg4 = 0 to 200 {
        %16 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg4)
        %17 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 5)>(%arg4)
        %18 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 5)>(%arg4)
        %19 = affine.apply affine_map<(d0, d1, d2) -> ((d2 floordiv 4) * 3 - ((d2 floordiv 4) floordiv 4) * 12 + d0 floordiv 3 + (d1 floordiv 8) floordiv 5)>(%arg3, %arg4, %arg2)
        %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 3 + d1 - (d0 floordiv 4) * 12 - (d1 floordiv 3) * 3 + d2 floordiv 8 - ((d2 floordiv 8) floordiv 5) * 5)>(%arg2, %arg3, %arg4)
        %21 = affine.load %alloc_16[%c0_2, %19, %20, %16] : memref<1x18x18x8xf32>
        %22 = affine.load %0[%8, %18, %17, %16] : memref<16x5x5x8xf32>
        %23 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %24 = arith.mulf %21, %22 : f32
        %25 = arith.addf %23, %24 : f32
        affine.store %25, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %10 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %11 = affine.load %1[%c0_11, %8, %c0_11, %c0_11] : memref<1x16x1x1xf32>
      %12 = affine.load %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %13 = arith.addf %10, %11 : f32
      %14 = arith.maximumf %13, %cst : f32
      %15 = arith.maximumf %12, %14 : f32
      affine.store %15, %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
    }
    %9 = affine.load %alloc_3[%c0_25, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.store %9, %alloc_18[%c0_25, %8, %7, %6] : memref<1x16x4x4xf32>
  }
  %collapse_shape_19 = memref.collapse_shape %alloc_18 [[0], [1, 2, 3]] : memref<1x16x4x4xf32> into memref<1x256xf32>
  %expand_shape_20 = memref.expand_shape %collapse_shape_19 [[0, 1], [2]] output_shape [1, 1, 256] : memref<1x256xf32> into memref<1x1x256xf32>
  %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %c10 = arith.constant 10 : index
  affine.parallel (%arg2) = (0) to (symbol(%c10)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%arg2)
    %c0_25 = arith.constant 0 : index
    %7 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg2)
    affine.store %cst, %alloc_21[%c0_0, %c0, %6] : memref<1x1x10xf32>
    affine.for %arg3 = 0 to 256 {
      %8 = affine.load %expand_shape_20[%7, %c0_25, %arg3] : memref<1x1x256xf32>
      %9 = affine.load %4[%7, %arg3, %6] : memref<1x256x10xf32>
      %10 = affine.load %alloc_21[%7, %c0_25, %6] : memref<1x1x10xf32>
      %11 = arith.mulf %8, %9 : f32
      %12 = arith.addf %10, %11 : f32
      affine.store %12, %alloc_21[%7, %c0_25, %6] : memref<1x1x10xf32>
    }
  }
  %collapse_shape_22 = memref.collapse_shape %alloc_21 [[0, 1], [2]] : memref<1x1x10xf32> into memref<1x10xf32>
  %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %c10_24 = arith.constant 10 : index
  affine.parallel (%arg2) = (0) to (symbol(%c10_24)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%arg2)
    %7 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg2)
    %8 = affine.load %collapse_shape_22[%c0_11, %6] : memref<1x10xf32>
    %9 = affine.load %5[%c0_11, %6] : memref<1x10xf32>
    %10 = arith.addf %8, %9 : f32
    affine.store %10, %alloc_23[%7, %6] : memref<1x10xf32>
  }
  memref.copy %alloc_23, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  return
}

