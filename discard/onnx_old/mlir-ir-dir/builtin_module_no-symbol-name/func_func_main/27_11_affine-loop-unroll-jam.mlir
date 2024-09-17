// -----// IR Dump After AffineLoopUnrollAndJam (affine-loop-unroll-jam) //----- //
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
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 1 {
          affine.store %cst, %alloc_14[%arg2, %arg3, %arg4, %arg5] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %subview = memref.subview %alloc_14[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %expand_shape, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst_12, %alloc_15[%c0_10, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %6 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              %7 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg4, %arg7)
              affine.store %cst, %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              %9 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg4, %arg7)
              affine.for %arg8 = 0 to 5 {
                affine.for %arg9 = 0 to 5 {
                  %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%8, %arg8)
                  %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%9, %arg9)
                  %20 = affine.load %alloc_14[%c0_9, %18, %19, %c0_8] : memref<1x32x32x1xf32>
                  %21 = affine.load %3[%arg5, %arg8, %arg9, %c0_8] : memref<8x5x5x1xf32>
                  %22 = affine.load %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
                  %23 = arith.mulf %20, %21 : f32
                  %24 = arith.addf %22, %23 : f32
                  affine.store %24, %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
                }
              }
              %10 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              %11 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg4, %arg7)
              %12 = affine.load %alloc_6[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %13 = affine.load %2[%c0_11, %arg5, %c0_11, %c0_11] : memref<1x8x1x1xf32>
              %14 = affine.load %alloc_15[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
              %15 = arith.addf %12, %13 : f32
              %16 = arith.maximumf %15, %cst : f32
              %17 = arith.maximumf %14, %16 : f32
              affine.store %17, %alloc_15[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
            }
          }
        }
      }
    }
  }
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 18 {
      affine.for %arg4 = 0 to 18 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst, %alloc_16[%arg2, %arg3, %arg4, %arg5] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %subview_17 = memref.subview %alloc_16[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_15, %subview_17 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 4 {
          affine.store %cst_12, %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
          affine.for %arg6 = 0 to 3 {
            affine.for %arg7 = 0 to 3 {
              %7 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg6)
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg5, %arg7)
              affine.store %cst, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %9 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg6)
              %10 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg5, %arg7)
              affine.for %arg8 = 0 to 5 {
                affine.for %arg9 = 0 to 5 {
                  affine.for %arg10 = 0 to 8 {
                    %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%9, %arg8)
                    %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%10, %arg9)
                    %21 = affine.load %alloc_16[%c0_2, %19, %20, %arg10] : memref<1x18x18x8xf32>
                    %22 = affine.load %0[%arg3, %arg8, %arg9, %arg10] : memref<16x5x5x8xf32>
                    %23 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
                    %24 = arith.mulf %21, %22 : f32
                    %25 = arith.addf %23, %24 : f32
                    affine.store %25, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
                  }
                }
              }
              %11 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg6)
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg5, %arg7)
              %13 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %14 = affine.load %1[%c0_11, %arg3, %c0_11, %c0_11] : memref<1x16x1x1xf32>
              %15 = affine.load %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
              %16 = arith.addf %13, %14 : f32
              %17 = arith.maximumf %16, %cst : f32
              %18 = arith.maximumf %15, %17 : f32
              affine.store %18, %alloc_3[0, 0, 0, 0] : memref<1x1x1x1xf32>
            }
          }
          %6 = affine.load %alloc_3[%arg2, 0, 0, 0] : memref<1x1x1x1xf32>
          affine.store %6, %alloc_18[%arg2, %arg3, %arg4, %arg5] : memref<1x16x4x4xf32>
        }
      }
    }
  }
  %collapse_shape_19 = memref.collapse_shape %alloc_18 [[0], [1, 2, 3]] : memref<1x16x4x4xf32> into memref<1x256xf32>
  %expand_shape_20 = memref.expand_shape %collapse_shape_19 [[0, 1], [2]] output_shape [1, 1, 256] : memref<1x256xf32> into memref<1x1x256xf32>
  %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 10 {
        affine.store %cst, %alloc_21[%c0_0, %c0, %arg4] : memref<1x1x10xf32>
        affine.for %arg5 = 0 to 256 {
          %6 = affine.load %expand_shape_20[%arg2, %arg3, %arg5] : memref<1x1x256xf32>
          %7 = affine.load %4[%arg2, %arg5, %arg4] : memref<1x256x10xf32>
          %8 = affine.load %alloc_21[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
          %9 = arith.mulf %6, %7 : f32
          %10 = arith.addf %8, %9 : f32
          affine.store %10, %alloc_21[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
        }
      }
    }
  }
  %collapse_shape_22 = memref.collapse_shape %alloc_21 [[0, 1], [2]] : memref<1x1x10xf32> into memref<1x10xf32>
  %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 10 {
      %6 = affine.load %collapse_shape_22[%c0_11, %arg3] : memref<1x10xf32>
      %7 = affine.load %5[%c0_11, %arg3] : memref<1x10xf32>
      %8 = arith.addf %6, %7 : f32
      affine.store %8, %alloc_23[%arg2, %arg3] : memref<1x10xf32>
    }
  }
  memref.copy %alloc_23, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  return
}

