// -----// IR Dump After SimplifyAffineStructures (affine-simplify-structures) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %alloc_2 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %c0_5 = arith.constant 0 : index
  %c0_6 = arith.constant 0 : index
  %c0_7 = arith.constant 0 : index
  %c0_8 = arith.constant 0 : index
  %alloc_9 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_10 = arith.constant 0 : index
  %c0_11 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_12 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %2 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %3 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %4 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %5 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %6 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %0, %alloc_13 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc_13 to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.parallel (%arg1) = (0) to (1) {
    affine.parallel (%arg2) = (0) to (32) {
      affine.parallel (%arg3) = (0) to (32) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.store %cst, %alloc_14[%arg1, %arg2, %arg3, %arg4] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %reinterpret_cast_15 = memref.reinterpret_cast %alloc_14 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_15 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.parallel (%arg1) = (0) to (1) {
    affine.for %arg2 = 0 to 1568 {
      %8 = affine.apply affine_map<(d0) -> (d0 mod 8)>(%arg2)
      %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) mod 14)>(%arg2)
      %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 8) floordiv 14)>(%arg2)
      affine.store %cst_12, %alloc_16[%c0_6, %10, %9, %8] : memref<1x14x14x8xf32>
      affine.for %arg3 = 0 to 4 {
        affine.store %cst, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %11 = affine.apply affine_map<(d0, d1) -> (((d1 floordiv 8) floordiv 14) * 2 + d0 floordiv 2)>(%arg3, %arg2)
        %12 = affine.apply affine_map<(d0, d1) -> (d0 + (d1 floordiv 8) * 2 - ((d1 floordiv 8) floordiv 14) * 28 - (d0 floordiv 2) * 2)>(%arg3, %arg2)
        affine.for %arg4 = 0 to 5 {
          %19 = affine.load %alloc_14[%c0_5, %11 + %arg4, %12 + %c0, %c0_4] : memref<1x32x32x1xf32>
          %20 = affine.load %4[%8, %arg4, %c0, %c0_4] : memref<8x5x5x1xf32>
          %21 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
          %22 = arith.mulf %19, %20 : f32
          %23 = arith.addf %21, %22 : f32
          affine.store %23, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
          %c1 = arith.constant 1 : index
          %24 = affine.load %alloc_14[%c0_5, %11 + %arg4, %12 + %c1, %c0_4] : memref<1x32x32x1xf32>
          %25 = affine.load %4[%8, %arg4, %c1, %c0_4] : memref<8x5x5x1xf32>
          %26 = arith.mulf %24, %25 : f32
          %27 = arith.addf %23, %26 : f32
          affine.store %27, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
          %c2 = arith.constant 2 : index
          %28 = affine.load %alloc_14[%c0_5, %11 + %arg4, %12 + %c2, %c0_4] : memref<1x32x32x1xf32>
          %29 = affine.load %4[%8, %arg4, %c2, %c0_4] : memref<8x5x5x1xf32>
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.addf %27, %30 : f32
          affine.store %31, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
          %c3 = arith.constant 3 : index
          %32 = affine.load %alloc_14[%c0_5, %11 + %arg4, %12 + %c3, %c0_4] : memref<1x32x32x1xf32>
          %33 = affine.load %4[%8, %arg4, %c3, %c0_4] : memref<8x5x5x1xf32>
          %34 = arith.mulf %32, %33 : f32
          %35 = arith.addf %31, %34 : f32
          affine.store %35, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
          %36 = affine.load %alloc_14[%c0_5, %11 + %arg4, %12 + %c4, %c0_4] : memref<1x32x32x1xf32>
          %37 = affine.load %4[%8, %arg4, %c4, %c0_4] : memref<8x5x5x1xf32>
          %38 = arith.mulf %36, %37 : f32
          %39 = arith.addf %35, %38 : f32
          affine.store %39, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
        }
        %13 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %14 = affine.load %3[0, %8, 0, 0] : memref<1x8x1x1xf32>
        %15 = affine.load %alloc_16[%arg1, %10, %9, %8] : memref<1x14x14x8xf32>
        %16 = arith.addf %13, %14 : f32
        %17 = arith.maximumf %16, %cst : f32
        %18 = arith.maximumf %15, %17 : f32
        affine.store %18, %alloc_16[%arg1, %10, %9, %8] : memref<1x14x14x8xf32>
      }
    }
  }
  %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.parallel (%arg1) = (0) to (1) {
    affine.parallel (%arg2) = (0) to (18) {
      affine.parallel (%arg3) = (0) to (18) {
        affine.parallel (%arg4) = (0) to (8) step (4) {
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %arg4] : memref<1x18x18x8xf32>
          %8 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg4)
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %8] : memref<1x18x18x8xf32>
          %9 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %9] : memref<1x18x18x8xf32>
          %10 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg4)
          affine.store %cst, %alloc_17[%arg1, %arg2, %arg3, %10] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %reinterpret_cast_18 = memref.reinterpret_cast %alloc_17 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_16, %reinterpret_cast_18 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.parallel (%arg1) = (0) to (1) {
    affine.for %arg2 = 0 to 256 {
      %8 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%arg2)
      %9 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) mod 4)>(%arg2)
      %10 = affine.apply affine_map<(d0) -> ((d0 floordiv 4) floordiv 4)>(%arg2)
      affine.store %cst_12, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.for %arg3 = 0 to 9 {
        affine.store %cst, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %12 = affine.apply affine_map<(d0, d1) -> ((d1 floordiv 4) * 3 - ((d1 floordiv 4) floordiv 4) * 12 + d0 floordiv 3)>(%arg3, %arg2)
        %13 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1 - (d0 floordiv 4) * 12 - (d1 floordiv 3) * 3)>(%arg2, %arg3)
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            affine.for %arg6 = 0 to 8 step 4 {
              %20 = affine.load %alloc_17[%c0_11, %12 + %arg4, %13 + %arg5, %arg6] : memref<1x18x18x8xf32>
              %21 = affine.load %1[%10, %arg4, %arg5, %arg6] : memref<16x5x5x8xf32>
              %22 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %23 = arith.mulf %20, %21 : f32
              %24 = arith.addf %22, %23 : f32
              affine.store %24, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %25 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg6)
              %26 = affine.load %alloc_17[%c0_11, %12 + %arg4, %13 + %arg5, %25] : memref<1x18x18x8xf32>
              %27 = affine.load %1[%10, %arg4, %arg5, %25] : memref<16x5x5x8xf32>
              %28 = arith.mulf %26, %27 : f32
              %29 = arith.addf %24, %28 : f32
              affine.store %29, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %30 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg6)
              %31 = affine.load %alloc_17[%c0_11, %12 + %arg4, %13 + %arg5, %30] : memref<1x18x18x8xf32>
              %32 = affine.load %1[%10, %arg4, %arg5, %30] : memref<16x5x5x8xf32>
              %33 = arith.mulf %31, %32 : f32
              %34 = arith.addf %29, %33 : f32
              affine.store %34, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %35 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg6)
              %36 = affine.load %alloc_17[%c0_11, %12 + %arg4, %13 + %arg5, %35] : memref<1x18x18x8xf32>
              %37 = affine.load %1[%10, %arg4, %arg5, %35] : memref<16x5x5x8xf32>
              %38 = arith.mulf %36, %37 : f32
              %39 = arith.addf %34, %38 : f32
              affine.store %39, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
            }
          }
        }
        %14 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %15 = affine.load %2[0, %10, 0, 0] : memref<1x16x1x1xf32>
        %16 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %17 = arith.addf %14, %15 : f32
        %18 = arith.maximumf %17, %cst : f32
        %19 = arith.maximumf %16, %18 : f32
        affine.store %19, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %11 = affine.load %alloc[%arg1, 0, 0, 0] : memref<1x1x1x1xf32>
      affine.store %11, %alloc_19[%arg1, %10, %9, %8] : memref<1x16x4x4xf32>
    }
  }
  %reinterpret_cast_20 = memref.reinterpret_cast %alloc_19 to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<1x16x4x4xf32> to memref<1x1x256xf32>
  %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.parallel (%arg1) = (0) to (1) {
    affine.parallel (%arg2) = (0) to (1) {
      affine.parallel (%arg3) = (0) to (10) {
        affine.store %cst, %alloc_21[%c0_8, %c0_7, %arg3] : memref<1x1x10xf32>
        affine.for %arg4 = 0 to 256 step 4 {
          %8 = affine.load %reinterpret_cast_20[%arg1, %arg2, %arg4] : memref<1x1x256xf32>
          %9 = affine.load %5[%arg1, %arg4, %arg3] : memref<1x256x10xf32>
          %10 = affine.load %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %11 = arith.mulf %8, %9 : f32
          %12 = arith.addf %10, %11 : f32
          affine.store %12, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %13 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg4)
          %14 = affine.load %reinterpret_cast_20[%arg1, %arg2, %13] : memref<1x1x256xf32>
          %15 = affine.load %5[%arg1, %13, %arg3] : memref<1x256x10xf32>
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %12, %16 : f32
          affine.store %17, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %18 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
          %19 = affine.load %reinterpret_cast_20[%arg1, %arg2, %18] : memref<1x1x256xf32>
          %20 = affine.load %5[%arg1, %18, %arg3] : memref<1x256x10xf32>
          %21 = arith.mulf %19, %20 : f32
          %22 = arith.addf %17, %21 : f32
          affine.store %22, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %23 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg4)
          %24 = affine.load %reinterpret_cast_20[%arg1, %arg2, %23] : memref<1x1x256xf32>
          %25 = affine.load %5[%arg1, %23, %arg3] : memref<1x256x10xf32>
          %26 = arith.mulf %24, %25 : f32
          %27 = arith.addf %22, %26 : f32
          affine.store %27, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
        }
      }
    }
  }
  %reinterpret_cast_22 = memref.reinterpret_cast %alloc_21 to offset: [0], sizes: [1, 10], strides: [10, 1] : memref<1x1x10xf32> to memref<1x10xf32>
  %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.parallel (%arg1) = (0) to (1) {
    affine.parallel (%arg2) = (0) to (8) step (4) {
      %8 = affine.load %reinterpret_cast_22[0, %arg2] : memref<1x10xf32>
      %9 = affine.load %6[0, %arg2] : memref<1x10xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %alloc_23[%arg1, %arg2] : memref<1x10xf32>
      %11 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg2)
      %12 = affine.load %reinterpret_cast_22[0, %11] : memref<1x10xf32>
      %13 = affine.load %6[0, %11] : memref<1x10xf32>
      %14 = arith.addf %12, %13 : f32
      affine.store %14, %alloc_23[%arg1, %11] : memref<1x10xf32>
      %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg2)
      %16 = affine.load %reinterpret_cast_22[0, %15] : memref<1x10xf32>
      %17 = affine.load %6[0, %15] : memref<1x10xf32>
      %18 = arith.addf %16, %17 : f32
      affine.store %18, %alloc_23[%arg1, %15] : memref<1x10xf32>
      %19 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg2)
      %20 = affine.load %reinterpret_cast_22[0, %19] : memref<1x10xf32>
      %21 = affine.load %6[0, %19] : memref<1x10xf32>
      %22 = arith.addf %20, %21 : f32
      affine.store %22, %alloc_23[%arg1, %19] : memref<1x10xf32>
    }
    affine.parallel (%arg2) = (8) to (10) {
      %8 = affine.load %reinterpret_cast_22[0, %arg2] : memref<1x10xf32>
      %9 = affine.load %6[0, %arg2] : memref<1x10xf32>
      %10 = arith.addf %8, %9 : f32
      affine.store %10, %alloc_23[%arg1, %arg2] : memref<1x10xf32>
    }
  }
  %7 = bufferization.to_tensor %alloc_23 : memref<1x10xf32>
  return %7 : tensor<1x10xf32>
}

