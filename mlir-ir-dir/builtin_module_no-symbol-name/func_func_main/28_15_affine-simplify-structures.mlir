// -----// IR Dump After SimplifyAffineStructures (affine-simplify-structures) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x10xf32> {
  %cst = arith.constant -3.40282347E+38 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %alloc_1 = memref.alloc() : memref<1x1x1x1xf32>
  %alloc_2 = memref.alloc() : memref<1x1x1x1xf32>
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc_3 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %collapse_shape = memref.collapse_shape %alloc_3 [[0, 1], [2], [3]] : memref<1x1x28x28xf32> into memref<1x28x28xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0], [1], [2, 3]] output_shape [1, 28, 28, 1] : memref<1x28x28xf32> into memref<1x28x28x1xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %c1024 = arith.constant 1024 : index
  affine.parallel (%arg1) = (0) to (symbol(%c1024)) {
    %c0_12 = arith.constant 0 : index
    %6 = affine.apply affine_map<(d0) -> (d0 mod 32)>(%arg1)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) mod 32)>(%arg1)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) floordiv 32)>(%arg1)
    affine.store %cst_0, %alloc_4[%8, %7, %6, %c0_12] : memref<1x32x32x1xf32>
  }
  %subview = memref.subview %alloc_4[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg1 = 0 to 1568 {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg1)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 14)>(%arg1)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 14)>(%arg1)
    %c0_12 = arith.constant 0 : index
    affine.store %cst, %alloc_5[%c0, %8, %7, %6] : memref<1x14x14x8xf32>
    affine.for %arg2 = 0 to 4 {
      affine.store %cst_0, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.for %arg3 = 0 to 25 {
        %15 = affine.apply affine_map<(d0) -> (d0 mod 5)>(%arg3)
        %16 = affine.apply affine_map<(d0) -> (d0 floordiv 5)>(%arg3)
        %17 = affine.apply affine_map<(d0, d1, d2) -> (((d2 floordiv 8) floordiv 14) * 2 + d1 floordiv 2 + d0 floordiv 5)>(%arg3, %arg2, %arg1)
        %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + (d2 floordiv 8) * 2 - ((d2 floordiv 8) floordiv 14) * 28 - (d1 floordiv 2) * 2 - (d0 floordiv 5) * 5)>(%arg3, %arg2, %arg1)
        %19 = affine.load %alloc_4[%c0, %17, %18, %c0] : memref<1x32x32x1xf32>
        %20 = affine.load %3[%6, %16, %15, %c0] : memref<8x5x5x1xf32>
        %21 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %22 = arith.mulf %19, %20 : f32
        %23 = arith.addf %21, %22 : f32
        affine.store %23, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %9 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %10 = affine.load %2[%c0, %6, %c0, %c0] : memref<1x8x1x1xf32>
      %11 = affine.load %alloc_5[%c0_12, %8, %7, %6] : memref<1x14x14x8xf32>
      %12 = arith.addf %9, %10 : f32
      %13 = arith.maximumf %12, %cst_0 : f32
      %14 = arith.maximumf %11, %13 : f32
      affine.store %14, %alloc_5[%c0_12, %8, %7, %6] : memref<1x14x14x8xf32>
    }
  }
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %c2592 = arith.constant 2592 : index
  affine.parallel (%arg1) = (0) to (symbol(%c2592)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg1)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 18)>(%arg1)
    %8 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 18) mod 18)>(%arg1)
    %9 = affine.apply affine_map<(d0) -> (((d0 floordiv 8) floordiv 18) floordiv 18)>(%arg1)
    affine.store %cst_0, %alloc_6[%9, %8, %7, %6] : memref<1x18x18x8xf32>
  }
  %subview_7 = memref.subview %alloc_6[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_5, %subview_7 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg1 = 0 to 256 {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg1)
    %7 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) mod 4)>(%arg1)
    %8 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) floordiv 4)>(%arg1)
    %c0_12 = arith.constant 0 : index
    affine.store %cst, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.for %arg2 = 0 to 9 {
      affine.store %cst_0, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.for %arg3 = 0 to 200 {
        %16 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg3)
        %17 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 5)>(%arg3)
        %18 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 5)>(%arg3)
        %19 = affine.apply affine_map<(d0, d1, d2) -> ((d2 floordiv 4) * 3 - ((d2 floordiv 4) floordiv 4) * 12 + d0 floordiv 3 + (d1 floordiv 8) floordiv 5)>(%arg2, %arg3, %arg1)
        %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 3 + d1 - (d0 floordiv 4) * 12 - (d1 floordiv 3) * 3 + d2 floordiv 8 - ((d2 floordiv 8) floordiv 5) * 5)>(%arg1, %arg2, %arg3)
        %21 = affine.load %alloc_6[%c0, %19, %20, %16] : memref<1x18x18x8xf32>
        %22 = affine.load %0[%8, %18, %17, %16] : memref<16x5x5x8xf32>
        %23 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %24 = arith.mulf %21, %22 : f32
        %25 = arith.addf %23, %24 : f32
        affine.store %25, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %10 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %11 = affine.load %1[%c0, %8, %c0, %c0] : memref<1x16x1x1xf32>
      %12 = affine.load %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %13 = arith.addf %10, %11 : f32
      %14 = arith.maximumf %13, %cst_0 : f32
      %15 = arith.maximumf %12, %14 : f32
      affine.store %15, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
    }
    %9 = affine.load %alloc_1[%c0_12, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.store %9, %alloc_8[%c0_12, %8, %7, %6] : memref<1x16x4x4xf32>
  }
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %c10 = arith.constant 10 : index
  affine.parallel (%arg1) = (0) to (symbol(%c10)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%arg1)
    %c0_12 = arith.constant 0 : index
    %7 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg1)
    affine.store %cst_0, %alloc_9[%c0, %c0, %6] : memref<1x1x10xf32>
    affine.for %arg2 = 0 to 256 {
      %8 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg1)
      %9 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%arg2)
      %10 = affine.apply affine_map<(d0) -> ((d0 mod 16) floordiv 4)>(%arg2)
      %11 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg2)
      %12 = affine.load %alloc_8[%8, %9, %10, %11] : memref<1x16x4x4xf32>
      %13 = affine.load %4[%7, %arg2, %6] : memref<1x256x10xf32>
      %14 = affine.load %alloc_9[%7, %c0_12, %6] : memref<1x1x10xf32>
      %15 = arith.mulf %12, %13 : f32
      %16 = arith.addf %14, %15 : f32
      affine.store %16, %alloc_9[%7, %c0_12, %6] : memref<1x1x10xf32>
    }
  }
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %c10_11 = arith.constant 10 : index
  affine.parallel (%arg1) = (0) to (symbol(%c10_11)) {
    %6 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%arg1)
    %7 = affine.apply affine_map<(d0) -> (d0 floordiv 10)>(%arg1)
    %8 = affine.load %alloc_9[%c0, %c0, %6] : memref<1x1x10xf32>
    %9 = affine.load %5[%c0, %6] : memref<1x10xf32>
    %10 = arith.addf %8, %9 : f32
    affine.store %10, %alloc_10[%7, %6] : memref<1x10xf32>
  }
  return %alloc_10 : memref<1x10xf32>
}

