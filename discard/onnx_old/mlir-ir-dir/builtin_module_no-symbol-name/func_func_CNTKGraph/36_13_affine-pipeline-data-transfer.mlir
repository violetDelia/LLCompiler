// -----// IR Dump After AffinePipelineDataTransfer (affine-pipeline-data-transfer) //----- //
func.func @CNTKGraph(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x10xf32>) {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_5 = arith.constant -3.40282347E+38 : f32
  %0 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %1 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %2 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %3 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %4 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %arg0, %alloc : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 1 {
          affine.store %cst, %alloc_6[%arg2, %arg3, %arg4, %arg5] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_7 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc : memref<1x1x28x28xf32>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 28 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst, %alloc_8[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 28 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 1 {
            affine.store %cst, %alloc_8[%c0_2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
            affine.for %arg7 = 0 to 5 {
              affine.for %arg8 = 0 to 5 {
                %6 = affine.load %alloc_6[%arg2, %arg3 + %arg7, %arg4 + %arg8, %arg6] : memref<1x32x32x1xf32>
                %7 = affine.load %3[%arg5, %arg7, %arg8, %arg6] : memref<8x5x5x1xf32>
                %8 = affine.load %alloc_8[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
                %9 = arith.mulf %6, %7 : f32
                %10 = arith.addf %8, %9 : f32
                affine.store %10, %alloc_8[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
              }
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_6 : memref<1x32x32x1xf32>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst_5, %alloc_9[%c0_4, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %6 = affine.load %alloc_8[0, %arg3 * 2 + %arg6, %arg4 * 2 + %arg7, %arg5] : memref<1x28x28x8xf32>
              %7 = affine.load %2[0, %arg5, 0, 0] : memref<1x8x1x1xf32>
              %8 = affine.load %alloc_9[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
              %9 = arith.addf %6, %7 : f32
              %10 = arith.maximumf %9, %cst : f32
              %11 = arith.maximumf %8, %10 : f32
              affine.store %11, %alloc_9[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_8 : memref<1x28x28x8xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 18 {
      affine.for %arg4 = 0 to 18 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst, %alloc_10[%arg2, %arg3, %arg4, %arg5] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %reinterpret_cast_11 = memref.reinterpret_cast %alloc_10 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_9, %reinterpret_cast_11 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_9 : memref<1x14x14x8xf32>
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 16 {
          affine.store %cst, %alloc_12[%c0_3, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
          affine.for %arg6 = 0 to 5 {
            affine.for %arg7 = 0 to 5 {
              affine.for %arg8 = 0 to 8 {
                %6 = affine.load %alloc_10[%arg2, %arg3 + %arg6, %arg4 + %arg7, %arg8] : memref<1x18x18x8xf32>
                %7 = affine.load %0[%arg5, %arg6, %arg7, %arg8] : memref<16x5x5x8xf32>
                %8 = affine.load %alloc_12[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
                %9 = arith.mulf %6, %7 : f32
                %10 = arith.addf %8, %9 : f32
                affine.store %10, %alloc_12[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
              }
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_10 : memref<1x18x18x8xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 16 {
          affine.store %cst_5, %alloc_13[%c0, %arg3, %arg4, %arg5] : memref<1x4x4x16xf32>
          affine.for %arg6 = 0 to 3 {
            affine.for %arg7 = 0 to 3 {
              %6 = affine.load %alloc_12[0, %arg3 * 3 + %arg6, %arg4 * 3 + %arg7, %arg5] : memref<1x14x14x16xf32>
              %7 = affine.load %1[0, %arg5, 0, 0] : memref<1x16x1x1xf32>
              %8 = affine.load %alloc_13[%arg2, %arg3, %arg4, %arg5] : memref<1x4x4x16xf32>
              %9 = arith.addf %6, %7 : f32
              %10 = arith.maximumf %9, %cst : f32
              %11 = arith.maximumf %8, %10 : f32
              affine.store %11, %alloc_13[%arg2, %arg3, %arg4, %arg5] : memref<1x4x4x16xf32>
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_12 : memref<1x14x14x16xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 4 {
          %6 = affine.load %alloc_13[%arg2, %arg4, %arg5, %arg3] : memref<1x4x4x16xf32>
          affine.store %6, %alloc_14[%arg2, %arg3, %arg4, %arg5] : memref<1x16x4x4xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_13 : memref<1x4x4x16xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 10 {
        affine.store %cst, %alloc_15[%c0_1, %c0_0, %arg4] : memref<1x1x10xf32>
        affine.for %arg5 = 0 to 256 {
          %6 = affine.load %alloc_14[symbol(%arg2) + symbol(%arg3), %arg5 floordiv 16, (%arg5 mod 16) floordiv 4, %arg5 mod 4] : memref<1x16x4x4xf32>
          %7 = affine.load %4[%arg2, %arg5, %arg4] : memref<1x256x10xf32>
          %8 = affine.load %alloc_15[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
          %9 = arith.mulf %6, %7 : f32
          %10 = arith.addf %8, %9 : f32
          affine.store %10, %alloc_15[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_14 : memref<1x16x4x4xf32>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 10 {
      %6 = affine.load %alloc_15[0, 0, %arg3] : memref<1x1x10xf32>
      %7 = affine.load %5[0, %arg3] : memref<1x10xf32>
      %8 = arith.addf %6, %7 : f32
      affine.store %8, %alloc_16[%arg2, %arg3] : memref<1x10xf32>
    }
  }
  memref.dealloc %alloc_15 : memref<1x1x10xf32>
  memref.copy %alloc_16, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  memref.dealloc %alloc_16 : memref<1x10xf32>
  return
}
