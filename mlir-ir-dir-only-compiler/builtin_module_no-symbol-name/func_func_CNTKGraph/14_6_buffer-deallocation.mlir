// -----// IR Dump After BufferDeallocation (buffer-deallocation) //----- //
func.func @CNTKGraph(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<1x10xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -3.40282347E+38 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %1 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %2 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %3 = memref.get_global @__constant_16x8x5x5xf32 : memref<16x8x5x5xf32>
  %4 = memref.get_global @__constant_8x1x5x5xf32 : memref<8x1x5x5xf32>
  %5 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x1xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 28 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 1 {
          %6 = affine.load %arg0[%arg2, %arg5, %arg3, %arg4] : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
          affine.store %6, %alloc[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x1xf32>
        }
      }
    }
  }
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<8x5x5x1xf32>
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 5 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 1 {
          %6 = affine.load %4[%arg2, %arg5, %arg3, %arg4] : memref<8x1x5x5xf32>
          affine.store %6, %alloc_1[%arg2, %arg3, %arg4, %arg5] : memref<8x5x5x1xf32>
        }
      }
    }
  }
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 1 {
          affine.store %cst_0, %alloc_2[%arg2, %arg3, %arg4, %arg5] : memref<1x32x32x1xf32>
        }
      }
    }
  }
  %subview = memref.subview %alloc_2[0, 2, 2, 0] [1, 28, 28, 1] [1, 1, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %alloc, %subview : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc : memref<1x28x28x1xf32>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 28 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst_0, %alloc_3[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 28 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 5 {
            affine.for %arg7 = 0 to 5 {
              affine.for %arg8 = 0 to 1 {
                %6 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg3, %arg6)
                %7 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %arg7)
                %8 = affine.load %alloc_2[%arg2, %6, %7, %arg8] : memref<1x32x32x1xf32>
                %9 = affine.load %alloc_1[%arg5, %arg6, %arg7, %arg8] : memref<8x5x5x1xf32>
                %10 = affine.load %alloc_3[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
                %11 = arith.mulf %8, %9 : f32
                %12 = arith.addf %10, %11 : f32
                affine.store %12, %alloc_3[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
              }
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_2 : memref<1x32x32x1xf32>
  memref.dealloc %alloc_1 : memref<8x5x5x1xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x8x28x28xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %6 = affine.load %alloc_3[%arg2, %arg4, %arg5, %arg3] : memref<1x28x28x8xf32>
          affine.store %6, %alloc_4[%arg2, %arg3, %arg4, %arg5] : memref<1x8x28x28xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_3 : memref<1x28x28x8xf32>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x8x28x28xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %6 = affine.load %alloc_4[%c0, %arg3, %arg4, %arg5] : memref<1x8x28x28xf32>
          %7 = affine.load %1[%c0, %arg3, %c0, %c0] : memref<1x8x1x1xf32>
          %8 = arith.addf %6, %7 : f32
          affine.store %8, %alloc_5[%arg2, %arg3, %arg4, %arg5] : memref<1x8x28x28xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_4 : memref<1x8x28x28xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x8x28x28xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 28 {
          %6 = affine.load %alloc_5[%c0, %arg3, %arg4, %arg5] : memref<1x8x28x28xf32>
          %7 = arith.maximumf %6, %cst_0 : f32
          affine.store %7, %alloc_6[%arg2, %arg3, %arg4, %arg5] : memref<1x8x28x28xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_5 : memref<1x8x28x28xf32>
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 28 {
      affine.for %arg4 = 0 to 28 {
        affine.for %arg5 = 0 to 8 {
          %6 = affine.load %alloc_6[%arg2, %arg5, %arg3, %arg4] : memref<1x8x28x28xf32>
          affine.store %6, %alloc_7[%arg2, %arg3, %arg4, %arg5] : memref<1x28x28x8xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_6 : memref<1x8x28x28xf32>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst, %alloc_8[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 2 {
              %6 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg3, %arg6)
              %7 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg4, %arg7)
              %8 = affine.load %alloc_7[%arg2, %6, %7, %arg5] : memref<1x28x28x8xf32>
              %9 = affine.load %alloc_8[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
              %10 = arith.maximumf %9, %8 : f32
              affine.store %10, %alloc_8[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_7 : memref<1x28x28x8xf32>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x8x14x14xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 14 {
          %6 = affine.load %alloc_8[%arg2, %arg4, %arg5, %arg3] : memref<1x14x14x8xf32>
          affine.store %6, %alloc_9[%arg2, %arg3, %arg4, %arg5] : memref<1x8x14x14xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_8 : memref<1x14x14x8xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 8 {
          %6 = affine.load %alloc_9[%arg2, %arg5, %arg3, %arg4] : memref<1x8x14x14xf32>
          affine.store %6, %alloc_10[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x8xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_9 : memref<1x8x14x14xf32>
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<16x5x5x8xf32>
  affine.for %arg2 = 0 to 16 {
    affine.for %arg3 = 0 to 5 {
      affine.for %arg4 = 0 to 5 {
        affine.for %arg5 = 0 to 8 {
          %6 = affine.load %3[%arg2, %arg5, %arg3, %arg4] : memref<16x8x5x5xf32>
          affine.store %6, %alloc_11[%arg2, %arg3, %arg4, %arg5] : memref<16x5x5x8xf32>
        }
      }
    }
  }
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 18 {
      affine.for %arg4 = 0 to 18 {
        affine.for %arg5 = 0 to 8 {
          affine.store %cst_0, %alloc_12[%arg2, %arg3, %arg4, %arg5] : memref<1x18x18x8xf32>
        }
      }
    }
  }
  %subview_13 = memref.subview %alloc_12[0, 2, 2, 0] [1, 14, 14, 8] [1, 1, 1, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_10, %subview_13 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_10 : memref<1x14x14x8xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 16 {
          affine.store %cst_0, %alloc_14[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 16 {
          affine.for %arg6 = 0 to 5 {
            affine.for %arg7 = 0 to 5 {
              affine.for %arg8 = 0 to 8 {
                %6 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg3, %arg6)
                %7 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg4, %arg7)
                %8 = affine.load %alloc_12[%arg2, %6, %7, %arg8] : memref<1x18x18x8xf32>
                %9 = affine.load %alloc_11[%arg5, %arg6, %arg7, %arg8] : memref<16x5x5x8xf32>
                %10 = affine.load %alloc_14[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
                %11 = arith.mulf %8, %9 : f32
                %12 = arith.addf %10, %11 : f32
                affine.store %12, %alloc_14[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
              }
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_12 : memref<1x18x18x8xf32>
  memref.dealloc %alloc_11 : memref<16x5x5x8xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x16x14x14xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 14 {
          %6 = affine.load %alloc_14[%arg2, %arg4, %arg5, %arg3] : memref<1x14x14x16xf32>
          affine.store %6, %alloc_15[%arg2, %arg3, %arg4, %arg5] : memref<1x16x14x14xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_14 : memref<1x14x14x16xf32>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x16x14x14xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 14 {
          %6 = affine.load %alloc_15[%c0, %arg3, %arg4, %arg5] : memref<1x16x14x14xf32>
          %7 = affine.load %0[%c0, %arg3, %c0, %c0] : memref<1x16x1x1xf32>
          %8 = arith.addf %6, %7 : f32
          affine.store %8, %alloc_16[%arg2, %arg3, %arg4, %arg5] : memref<1x16x14x14xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_15 : memref<1x16x14x14xf32>
  %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x16x14x14xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 14 {
          %6 = affine.load %alloc_16[%c0, %arg3, %arg4, %arg5] : memref<1x16x14x14xf32>
          %7 = arith.maximumf %6, %cst_0 : f32
          affine.store %7, %alloc_17[%arg2, %arg3, %arg4, %arg5] : memref<1x16x14x14xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_16 : memref<1x16x14x14xf32>
  %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 14 {
      affine.for %arg4 = 0 to 14 {
        affine.for %arg5 = 0 to 16 {
          %6 = affine.load %alloc_17[%arg2, %arg5, %arg3, %arg4] : memref<1x16x14x14xf32>
          affine.store %6, %alloc_18[%arg2, %arg3, %arg4, %arg5] : memref<1x14x14x16xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_17 : memref<1x16x14x14xf32>
  %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 16 {
          affine.store %cst, %alloc_19[%arg2, %arg3, %arg4, %arg5] : memref<1x4x4x16xf32>
        }
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 16 {
          affine.for %arg6 = 0 to 3 {
            affine.for %arg7 = 0 to 3 {
              %6 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg3, %arg6)
              %7 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%arg4, %arg7)
              %8 = affine.load %alloc_18[%arg2, %6, %7, %arg5] : memref<1x14x14x16xf32>
              %9 = affine.load %alloc_19[%arg2, %arg3, %arg4, %arg5] : memref<1x4x4x16xf32>
              %10 = arith.maximumf %9, %8 : f32
              affine.store %10, %alloc_19[%arg2, %arg3, %arg4, %arg5] : memref<1x4x4x16xf32>
            }
          }
        }
      }
    }
  }
  memref.dealloc %alloc_18 : memref<1x14x14x16xf32>
  %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 4 {
          %6 = affine.load %alloc_19[%arg2, %arg4, %arg5, %arg3] : memref<1x4x4x16xf32>
          affine.store %6, %alloc_20[%arg2, %arg3, %arg4, %arg5] : memref<1x16x4x4xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_19 : memref<1x4x4x16xf32>
  %collapse_shape = memref.collapse_shape %alloc_20 [[0], [1, 2, 3]] : memref<1x16x4x4xf32> into memref<1x256xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2]] output_shape [1, 1, 256] : memref<1x256xf32> into memref<1x1x256xf32>
  %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 10 {
        affine.store %cst_0, %alloc_21[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
      }
    }
  }
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 10 {
        affine.for %arg5 = 0 to 256 {
          %6 = affine.load %expand_shape[%arg2, %arg3, %arg5] : memref<1x1x256xf32>
          %7 = affine.load %2[%arg2, %arg5, %arg4] : memref<1x256x10xf32>
          %8 = affine.load %alloc_21[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
          %9 = arith.mulf %6, %7 : f32
          %10 = arith.addf %8, %9 : f32
          affine.store %10, %alloc_21[%arg2, %arg3, %arg4] : memref<1x1x10xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_20 : memref<1x16x4x4xf32>
  %collapse_shape_22 = memref.collapse_shape %alloc_21 [[0, 1], [2]] : memref<1x1x10xf32> into memref<1x10xf32>
  %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  affine.for %arg2 = 0 to 1 {
    affine.for %arg3 = 0 to 10 {
      %6 = affine.load %collapse_shape_22[%c0, %arg3] : memref<1x10xf32>
      %7 = affine.load %5[%c0, %arg3] : memref<1x10xf32>
      %8 = arith.addf %6, %7 : f32
      affine.store %8, %alloc_23[%arg2, %arg3] : memref<1x10xf32>
    }
  }
  memref.dealloc %alloc_21 : memref<1x1x10xf32>
  memref.copy %alloc_23, %arg1 : memref<1x10xf32> to memref<1x10xf32>
  memref.dealloc %alloc_23 : memref<1x10xf32>
  return
}

