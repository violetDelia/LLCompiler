within split at /home/lfr/LLCompiler/third_party/llvm-project/mlir/test/Dialect/Affine/dma-generate.mlir:57 offset :65:16: error: unexpected error: operation being parsed with an unregistered dialect. If this is intended, please use -allow-unregistered-dialect with the MLIR tool used
          "foo"(%v0) : (f32) -> ()
               ^
module {
}

// -----
#map = affine_map<(d0) -> (d0 + 256)>
module {
  func.func @loop_nest_1d() {
    %c256 = arith.constant 256 : index
    %c256_0 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c256_1 = arith.constant 256 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<512xf32>
    %alloc_5 = memref.alloc() : memref<256xf32, 2>
    %alloc_6 = memref.alloc() : memref<256xf32, 2>
    %alloc_7 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c0_2], %alloc_6[%c0_2], %alloc_7[%c0_2], %c256_1 : memref<256xf32>, memref<256xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_7[%c0_2], %c256_1 : memref<1xi32>
    %alloc_8 = memref.alloc() : memref<256xf32, 2>
    %alloc_9 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc_4[%c256], %alloc_8[%c0], %alloc_9[%c0], %c256_0 : memref<512xf32>, memref<256xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_9[%c0], %c256_0 : memref<1xi32>
    affine.for %arg0 = 0 to 256 {
      %0 = affine.load %alloc_6[%arg0] : memref<256xf32, 2>
      %1 = affine.apply #map(%arg0)
      %2 = affine.load %alloc_8[%arg0] : memref<256xf32, 2>
      %3 = affine.load %alloc_5[%arg0] : memref<256xf32, 2>
    }
    memref.dealloc %alloc_9 : memref<1xi32>
    memref.dealloc %alloc_8 : memref<256xf32, 2>
    memref.dealloc %alloc_7 : memref<1xi32>
    memref.dealloc %alloc_6 : memref<256xf32, 2>
    return
  }
}

// -----
// -----
#map = affine_map<(d0) -> (d0 mod 2)>
module {
  func.func @loop_nest_modulo() {
    %c58 = arith.constant 58 : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256x8xf32>
    %alloc_1 = memref.alloc() : memref<29x2xf32, 2>
    %alloc_2 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c0, %c0], %alloc_1[%c0, %c0], %alloc_2[%c0], %c58, %c8, %c2 : memref<256x8xf32>, memref<29x2xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_2[%c0], %c58 : memref<1xi32>
    affine.for %arg0 = 0 to 32 step 4 {
      affine.for %arg1 = 0 to 8 {
        %0 = affine.apply #map(%arg1)
        %1 = affine.load %alloc_1[%arg0, %arg1 mod 2] : memref<29x2xf32, 2>
      }
    }
    memref.dealloc %alloc_2 : memref<1xi32>
    memref.dealloc %alloc_1 : memref<29x2xf32, 2>
    return
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 + 32)>
module {
  func.func @loop_nest_tiled() -> memref<256x1024xf32> {
    %c1024 = arith.constant 1024 : index
    %c1024_0 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256x1024xf32>
    affine.for %arg0 = 0 to 256 step 32 {
      affine.for %arg1 = 0 to 1024 step 32 {
        %0 = affine.apply #map(%arg0, %arg1)
        %1 = affine.apply #map1(%arg0, %arg1)
        %alloc_2 = memref.alloc() : memref<32x32xf32, 2>
        %alloc_3 = memref.alloc() : memref<1xi32>
        affine.dma_start %alloc[%arg0, %arg1], %alloc_2[%c0, %c0], %alloc_3[%c0], %c1024, %c1024_0, %c32 : memref<256x1024xf32>, memref<32x32xf32, 2>, memref<1xi32>
        affine.dma_wait %alloc_3[%c0], %c1024 : memref<1xi32>
        affine.for %arg2 = #map2(%arg0) to #map3(%arg0) {
          affine.for %arg3 = #map2(%arg1) to #map3(%arg1) {
            %2 = affine.load %alloc_2[-%arg0 + %arg2, -%arg1 + %arg3] : memref<32x32xf32, 2>
          }
        }
        memref.dealloc %alloc_3 : memref<1xi32>
        memref.dealloc %alloc_2 : memref<32x32xf32, 2>
      }
    }
    return %alloc : memref<256x1024xf32>
  }
}

// -----
module {
  func.func @dma_constant_dim_access(%arg0: memref<100x100xf32>) {
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c100within split at /home/lfr/LLCompiler/third_party/llvm-project/mlir/test/Dialect/Affine/dma-generate.mlir:305 offset :14:5: error: unexpected warning: total size of all copy buffers' for this block exceeds fast memory capacity
    affine.for %j = 0 to 256 {
    ^
_2 = arith.constant 100 : index
    %alloc = memref.alloc() : memref<1x100xf32, 2>
    %alloc_3 = memref.alloc() : memref<1xi32>
    affine.dma_start %arg0[%c1, %c0], %alloc[%c0, %c0], %alloc_3[%c0], %c100 : memref<100x100xf32>, memref<1x100xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_3[%c0], %c100 : memref<1xi32>
    affine.for %arg1 = 0 to 100 {
      affine.for %arg2 = 0 to %c100_2 {
        %0 = affine.load %alloc[0, %arg2] : memref<1x100xf32, 2>
      }
    }
    memref.dealloc %alloc_3 : memref<1xi32>
    memref.dealloc %alloc : memref<1x100xf32, 2>
    return
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1 + 9)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (d1 + s0 + s1)>
module {
  func.func @dma_with_symbolic_accesses(%arg0: memref<100x100xf32>, %arg1: index) {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    affine.for %arg2 = 0 to 100 {
      %0 = affine.apply #map(%arg2, %arg1)
      %1 = affine.apply #map1(%arg2, %arg1)
      %alloc = memref.alloc() : memref<1x100xf32, 2>
      %alloc_1 = memref.alloc() : memref<1xi32>
      affine.dma_start %arg0[%arg2, symbol(%arg1) + 9], %alloc[%c0, %c0], %alloc_1[%c0], %c100 : memref<100x100xf32>, memref<1x100xf32, 2>, memref<1xi32>
      affine.dma_wait %alloc_1[%c0], %c100 : memref<1xi32>
      affine.for %arg3 = 0 to 100 {
        %2 = affine.apply #map2(%arg2, %arg3)[%arg1, %c9]
        %3 = affine.load %alloc[0, %arg3] : memref<1x100xf32, 2>
      }
      memref.dealloc %alloc_1 : memref<1xi32>
      memref.dealloc %alloc : memref<1x100xf32, 2>
    }
    return
  }
}

// -----
#map = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func @dma_with_symbolic_loop_bounds(%arg0: memref<100x100xf32>, %arg1: index, %arg2: index) {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    affine.for %arg3 = 0 to 100 {
      %0 = affine.apply #map(%arg3, %arg1, %arg2)
      %alloc = memref.alloc() : memref<1x100xf32, 2>
      %alloc_1 = memref.alloc() : memref<1xi32>
      affine.dma_start %arg0[%arg3, 0], %alloc[%c0, %c0], %alloc_1[%c0], %c100 : memref<100x100xf32>, memref<1x100xf32, 2>, memref<1xi32>
      affine.dma_wait %alloc_1[%c0], %c100 : memref<1xi32>
      affine.for %arg4 = %arg1 to %arg2 {
        %1 = affine.apply #map1(%arg4)[%c9]
        %2 = affine.load %alloc[0, %arg4 + 9] : memref<1x100xf32, 2>
      }
      memref.dealloc %alloc_1 : memref<1xi32>
      memref.dealloc %alloc : memref<1x100xf32, 2>
    }
    return
  }
}

// -----
module {
  func.func @dma_unknown_size(%arg0: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0_1 : memref<?x?xf32>
    %dim_2 = memref.dim %arg0, %c0_1 : memref<?x?xf32>
    affine.for %arg1 = 0 to %dim {
      affine.for %arg2 = 0 to %dim_2 {
        %0 = affine.load %arg0[%arg1, %arg2] : memref<?x?xf32>
      }
    }
    return
  }
}

// -----
#map = affine_map<(d0) -> (d0 mod 128)>
module {
  func.func @dma_memref_3d(%arg0: memref<1024x1024x1024xf32>) {
    %c2097152 = arith.constant 2097152 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    affine.for %arg1 = 0 to 1024 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 1024 {
          %0 = affine.apply #map(%arg1)
          %1 = affine.apply #map(%arg2)
          %2 = affine.apply #map(%arg3)
          %alloc = memref.alloc() : memref<128x128x128xf32, 2>
          %3 = affine.load %arg0[%0, %1, %2] : memref<1024x1024x1024xf32>
        }
      }
    }
    return
  }
}

// -----
#map = affine_map<(d0) -> (d0 + 64)>
#map1 = affine_map<(d0) -> (d0 + 128)>
#map2 = affine_map<(d0) -> (d0 + 2)>
#map3 = affine_map<(d0, d1) -> (d0 + 2)>
#map4 = affine_map<(d0, d1) -> (d1 + 2)>
#map5 = affine_map<(d0) -> (d0 + 192)>
module {
  func.func @multi_load_store_union() {
    %c24257 = arith.constant 24257 : index
    %c512 = arith.constant 512 : index
    %c191 = arith.constant 191 : index
    %c0 = arith.constant 0 : index
    %c24257_0 = arith.constant 24257 : index
    %c512_1 = arith.constant 512 : index
    %c191_2 = arith.constant 191 : index
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<512x512xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 256 {
        %0 = affine.apply #map(%arg0)
        %1 = affine.apply #map1(%arg1)
        %2 = affine.apply #map2(%arg0)
        %3 = affine.apply #map2(%arg1)
        %4 = affine.apply #map3(%arg0, %arg1)
        %5 = affine.apply #map4(%arg0, %arg1)
        %alloc_5 = memref.alloc() : memref<127x191xf32, 2>
        %alloc_6 = memref.alloc() : memref<1xi32>
        affine.dma_start %alloc[%arg0 + 2, %arg1 + 2], %alloc_5[%c0_3, %c0_3], %alloc_6[%c0_3], %c24257_0, %c512_1, %c191_2 : memref<512x512xf32>, memref<127x191xf32, 2>, memref<1xi32>
        affine.dma_wait %alloc_6[%c0_3], %c24257_0 : memref<1xi32>
        %alloc_7 = memref.alloc() : memref<1xi32>
        %6 = affine.load %alloc_5[0, 126] : memref<127x191xf32, 2>
        %7 = affine.load %alloc_5[62, 0] : memref<127x191xf32, 2>
        %8 = affine.apply #map1(%arg0)
        %9 = affine.apply #map5(%arg1)
        affine.store %6, %alloc_5[0, 190] : memref<127x191xf32, 2>
        affine.store %7, %alloc_5[126, 0] : memref<127x191xf32, 2>
        %10 = affine.apply #map3(%arg0, %arg1)
        %11 = affine.apply #map4(%arg0, %arg1)
        affine.dma_start %alloc_5[%c0, %c0], %alloc[%arg0 + 2, %arg1 + 2], %alloc_7[%c0], %c24257, %c512, %c191 : memref<127x191xf32, 2>, memref<512x512xf32>, memref<1xi32>
        affine.dma_wait %alloc_7[%c0], %c24257 : memref<1xi32>
        memref.dealloc %alloc_7 : memref<1xi32>
        memref.dealloc %alloc_6 : memref<1xi32>
        memref.dealloc %alloc_5 : memref<127x191xf32, 2>
      }
    }
    return
  }
}

// -----
module {
  func.func @dma_loop_straightline_interspersed() {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c256_0 = arith.constant 256 : index
    %c0_1 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c254 = arith.constant 254 : index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    %c255 = arith.constant 255 : index
    %alloc = memref.alloc() : memref<256xf32>
    %alloc_7 = memref.alloc() : memref<1xf32, 2>
    %alloc_8 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c0_4], %alloc_7[%c0_4], %alloc_8[%c0_4], %c1_3 : memref<256xf32>, memref<1xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_8[%c0_4], %c1_3 : memref<1xi32>
    %0 = affine.load %alloc_7[0] : memref<1xf32, 2>
    memref.dealloc %alloc_8 : memref<1xi32>
    memref.dealloc %alloc_7 : memref<1xf32, 2>
    %alloc_9 = memref.alloc() : memref<254xf32, 2>
    %alloc_10 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c1], %alloc_9[%c0_2], %alloc_10[%c0_2], %c254 : memref<256xf32>, memref<254xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_10[%c0_2], %c254 : memref<1xi32>
    affine.for %arg0 = 1 to 255 {
      %2 = affine.load %alloc_9[%arg0 - 1] : memref<254xf32, 2>
    }
    memref.dealloc %alloc_10 : memref<1xi32>
    memref.dealloc %alloc_9 : memref<254xf32, 2>
    %alloc_11 = memref.alloc() : memref<256xf32, 2>
    %alloc_12 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c0_1], %alloc_11[%c0_1], %alloc_12[%c0_1], %c256_0 : memref<256xf32>, memref<256xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_12[%c0_1], %c256_0 : memref<1xi32>
    %alloc_13 = memref.alloc() : memref<1xi32>
    %1 = affine.load %alloc_11[255] : memref<256xf32, 2>
    affine.store %1, %alloc_11[0] : memref<256xf32, 2>
    affine.dma_start %alloc_11[%c0], %alloc[%c0], %alloc_13[%c0], %c256 : memref<256xf32, 2>, memref<256xf32>, memref<1xi32>
    affine.dma_wait %alloc_13[%c0], %c25within split at /home/lfr/LLCompiler/third_party/llvm-project/mlir/test/Dialect/Affine/dma-generate.mlir:406 offset :9:10: error: unexpected error: operation being parsed with an unregistered dialect. If this is intended, please use -allow-unregistered-dialect with the MLIR tool used
    "foo"(%v) : (vector<8 x f32>) -> ()
         ^
6 : memref<1xi32>
    memref.dealloc %alloc_13 : memref<1xi32>
    memref.dealloc %alloc_12 : memref<1xi32>
    memref.dealloc %alloc_11 : memref<256xf32, 2>
    return
  }
}

// -----
// -----
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 4)>
module {
  func.func @relative_loop_bounds(%arg0: memref<1027xf32>) {
    %c1027 = arith.constant 1027 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1027xf32, 2>
    %alloc_1 = memref.alloc() : memref<1xi32>
    affine.for %arg1 = 0 to 1024 {
      affine.for %arg2 = #map(%arg1) to #map1(%arg1) {
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %alloc[%arg2] : memref<1027xf32, 2>
      }
    }
    affine.dma_start %alloc[%c0], %arg0[%c0], %alloc_1[%c0], %c1027 : memref<1027xf32, 2>, memref<1027xf32>, memref<1xi32>
    affine.dma_wait %alloc_1[%c0], %c1027 : memref<1xi32>
    memref.dealloc %alloc_1 : memref<1xi32>
    memref.dealloc %alloc : memref<1027xf32, 2>
    return
  }
}

// -----
#map = affine_map<(d0) -> (d0 + 100)>
#map1 = affine_map<(d0) -> (d0 + 25)>
module {
  func.func @test_read_write_region_union() {
    %c25 = arith.constant 25 : index
    %c85 = arith.constant 85 : index
    %c0 = arith.constant 0 : index
    %c25_0 = arith.constant 25 : index
    %c85_1 = arith.constant 85 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256xf32>
    %alloc_4 = memref.alloc() : memref<85xf32, 2>
    %alloc_5 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c25_0], %alloc_4[%c0_2], %alloc_5[%c0_2], %c85_1 : memref<256xf32>, memref<85xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_5[%c0_2], %c85_1 : memref<1xi32>
    %alloc_6 = memref.alloc() : memref<1xi32>
    affine.for %arg0 = 0 to 10 {
      %0 = affine.apply #map(%arg0)
      %1 = affine.apply #map1(%arg0)
      %2 = affine.load %alloc_4[%arg0 + 75] : memref<85xf32, 2>
      affine.store %2, %alloc_4[%arg0] : memref<85xf32, 2>
    }
    affine.dma_start %alloc_4[%c0], %alloc[%c25], %alloc_6[%c0], %c85 : memref<85xf32, 2>, memref<256xf32>, memref<1xi32>
    affine.dma_wait %alloc_6[%c0], %c85 : memref<1xi32>
    memref.dealloc %alloc_6 : memref<1xi32>
    memref.dealloc %alloc_5 : memref<1xi32>
    memref.dealloc %alloc_4 : memref<85xf32, 2>
    return
  }
}

// -----
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 3)>
#map2 = affine_map<(d0) -> (d0 floordiv 8)>
module {
  func.func @test_analysis_util(%arg0: memref<4x4x16x1xf32>, %arg1: memref<144x9xf32>, %arg2: memref<2xf32>) -> (memref<144x9xf32>, memref<2xf32>) {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c0_0 = arith.constant 0 : index
    %c2_1 = arith.constant 2 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<64x1xf32>
    %alloc_5 = memref.alloc() : memref<144x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %alloc_6 = memref.alloc() : memref<2xf32, 2>
    %alloc_7 = memref.alloc() : memref<1xi32>
    affine.dma_start %arg2[%c0_2], %alloc_6[%c0_2], %alloc_7[%c0_2], %c2_1 : memref<2xf32>, memref<2xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_7[%c0_2], %c2_1 : memref<1xi32>
    %alloc_8 = memref.alloc() : memref<64x1xf32, 2>
    %alloc_9 = memref.alloc() : memref<1xi32>
    affine.dma_start %alloc[%c0_0, %c0_0], %alloc_8[%c0_0, %c0_0], %alloc_9[%c0_0], %c64 : memref<64x1xf32>, memref<64x1xf32, 2>, memref<1xi32>
    affine.dma_wait %alloc_9[%c0_0], %c64 : memref<1xi32>
    %alloc_10 = memref.alloc() : memref<1xi32>
    affine.for %arg3 = 0 to 9 step 3 {
      affine.for %arg4 = #map(%arg3) to #map1(%arg3) {
        affine.for %arg5 = 0 to 64 {
          %0 = affine.apply #map2(%arg4)
          %1 = affine.load %alloc_6[%arg4 floordiv 8] : memref<2xf32, 2>
          %2 = affine.apply #map(%arg5)
          %3 = affine.load %alloc_8[%arg5, 0] : memref<64x1xf32, 2>
         within split at /home/lfr/LLCompiler/third_party/llvm-project/mlir/test/Dialect/Affine/dma-generate.mlir:516 offset :14:7: error: unexpected warning: total size of all copy buffers' for this block exceeds fast memory capacity
      affine.for %i10 = 0 to 64 {
      ^
 affine.store %3, %alloc_6[%arg4 floordiv 8] : memref<2xf32, 2>
        }
      }
    }
    affine.dma_start %alloc_6[%c0], %arg2[%c0], %alloc_10[%c0], %c2 : memref<2xf32, 2>, memref<2xf32>, memref<1xi32>
    affine.dma_wait %alloc_10[%c0], %c2 : memref<1xi32>
    memref.dealloc %alloc_10 : memref<1xi32>
    memref.dealloc %alloc_9 : memref<1xi32>
    memref.dealloc %alloc_8 : memref<64x1xf32, 2>
    memref.dealloc %alloc_7 : memref<1xi32>
    memref.dealloc %alloc_6 : memref<2xf32, 2>
    return %arg1, %arg2 : memref<144x9xf32>, memref<2xf32>
  }
}

// -----
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 3)>
#map2 = affine_map<(d0, d1) -> ((d0 + d1 * 72) floordiv 2304 + (d0 mod 9) floordiv 3)>
#map3 = affine_map<(d0, d1) -> ((d0 + d1 * 72) mod 2304 - (((d0 + d1 * 72) mod 2304) floordiv 1152) * 1151 - (((d0 + d1 * 72) mod 1152) floordiv 9) * 9 - ((d0 mod 9) floordiv 3) * 3)>
#map4 = affine_map<(d0, d1) -> ((((d0 + d1 * 72) mod 1152) floordiv 9) floordiv 8)>
module {
  func.func @test_memref_bounds(%arg0: memref<4x4x16x1xvector<8x128xf32>>, %arg1: memref<144x9xvector<8x128xf32>>, %arg2: memref<2xvector<8x128xf32>>) -> (memref<144x9xvector<8x128xf32>>, memref<2xvector<8x128xf32>>) {
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    affine.for %arg3 = 0 to 9 step 3 {
      affine.for %arg4 = #map(%arg3) to #map1(%arg3) {
        affine.for %arg5 = 0 to 64 {
          %0 = affine.apply #map2(%arg4, %arg5)
          %1 = affine.apply #map3(%arg4, %arg5)
          %2 = affine.apply #map4(%arg4, %arg5)
          %alloc = memref.alloc() : memref<4x4x16x1xvector<8x128xf32>, 2>
          %alloc_2 = memref.alloc() : memref<1xi32>
          affine.dma_start %arg0[%c0, %c0, %c0, %c0], %alloc[%c0, %c0, %c0, %c0], %alloc_2[%c0], %c256 : memref<4x4x16x1xvector<8x128xf32>>, memref<4x4x16x1xvector<8x128xf32>, 2>, memref<1xi32>
          affine.dma_wait %alloc_2[%c0], %c256 : memref<1xi32>
          %3 = affine.load %alloc[(%arg4 + %arg5 * 72) floordiv 2304 + (%arg4 mod 9) floordiv 3, (%arg4 + %arg5 * 72) mod 2304 - (((%arg4 + %arg5 * 72) mod 2304) floordiv 1152) * 1151 - (((%arg4 + %arg5 * 72) mod 1152) floordiv 9) * 9 - ((%arg4 mod 9) floordiv 3) * 3, (((%arg4 + %arg5 * 72) mod 1152) floordiv 9) floordiv 8, 0] : memref<4x4x16x1xvector<8x128xf32>, 2>
          memref.dealloc %alloc_2 : memref<1xi32>
          memref.dealloc %alloc : memref<4x4x16x1xvector<8x128xf32>, 2>
        }
      }
    }
    return %arg1, %arg2 : memref<144x9xvector<8x128xf32>>, memref<2xvector<8x128xf32>>
  }
}

// -----
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 4)>
module {
  func.func @load_store_same_memref(%arg0: memref<256x1024xf32>) {
    %c4096 = arith.constant 4096 : index
    %c0 = arith.constant 0 : index
    %c4096_0 = arith.constant 4096 : index
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    affine.for %arg1 = 0 to 256 step 4 {
      %0 = affine.apply #map(%arg1)
      %alloc = memref.alloc() : memref<4x1024xf32, 2>
      %alloc_3 = memref.alloc() : memref<1xi32>
      affine.dma_start %arg0[%arg1, 0], %alloc[%c0_1, %c0_1], %alloc_3[%c0_1], %c4096_0 : memref<256x1024xf32>, memref<4x1024xf32, 2>, memref<1xi32>
      affine.dma_wait %alloc_3[%c0_1], %c4096_0 : memref<1xi32>
      %alloc_4 = memref.alloc() : memref<1xi32>
      affine.for %arg2 = 0 to 1024 step 4 {
        affine.for %arg3 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
            %2 = affine.load %alloc[-%arg1 + %arg3, %arg4] : memref<4x1024xf32, 2>
            %3 = arith.mulf %2, %2 : f32
            affine.store %3, %alloc[-%arg1 + %arg3, %arg4] : memref<4x1024xf32, 2>
          }
        }
      }
      %1 = affine.apply #map(%arg1)
      affine.dma_start %alloc[%c0, %c0], %arg0[%arg1, 0], %alloc_4[%c0], %c4096 : memref<4x1024xf32, 2>, memref<256x1024xf32>, memref<1xi32>
      affine.dma_wait %alloc_4[%c0], %c4096 : memref<1xi32>
      memref.dealloc %alloc_4 : memref<1xi32>
      memref.dealloc %alloc_3 : memref<1xi32>
      memref.dealloc %alloc : memref<4x1024xf32, 2>
    }
    return
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> (d0 + 4)>
module {
  func.func @simple_matmul(%arg0: memref<8x8xvector<64xf32>>, %arg1: memref<8x8xvector<64xf32>>, %arg2: memref<8x8xvector<64xf32>>) -> memref<8x8xvector<64xf32>> {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c16_0 = arith.constant 16 : index
    %c8_1 = arith.constant 8 : index
    %c4_2 = arith.constant 4 : index
    %c0_3 = arith.constant 0 : index
    %c16_4 = arith.constant 16 : index
    %c8_5 = arith.constant 8 : index
    %c4_6 = arith.constant 4 : index
    %c0_7 = arith.constant 0 : index
    %c16_8 = arith.constant 16 : index
    %c8_9 = arith.constant 8 : index
    %c4_10 = arith.constant 4 : index
    %c0_11 = arith.constant 0 : index
    %c0_12 = arith.constant 0 : index
    affine.for %arg3 = 0 to 8 step 4 {
      affine.for %arg4 = 0 to 8 step 4 {
        %0 = affine.apply #map(%arg3, %arg4)
        %1 = affine.apply #map1(%arg3, %arg4)
        %alloc = memref.alloc() : memref<4x4xvector<64xf32>, 2>
        %alloc_13 = memref.alloc() : memref<1xi32>
        affine.dma_start %arg2[%arg3, %arg4], %alloc[%c0_3, %c0_3], %alloc_13[%c0_3], %c16_0, %c8_1, %c4_2 : memref<8x8xvector<64xf32>>, memref<4x4xvector<64xf32>, 2>, memref<1xi32>
        affine.dma_wait %alloc_13[%c0_3], %c16_0 : memref<1xi32>
        %alloc_14 = memref.alloc() : memref<1xi32>
        affine.for %arg5 = 0 to 8 step 4 {
          %4 = affine.apply #map(%arg3, %arg5)
          %5 = affine.apply #map1(%arg3, %arg5)
          %alloc_15 = memref.alloc() : memref<4x4xvector<64xf32>, 2>
          %alloc_16 = memref.alloc() : memref<1xi32>
          affine.dma_start %arg0[%arg3, %arg5], %alloc_15[%c0_11, %c0_11], %alloc_16[%c0_11], %c16_8, %c8_9, %c4_10 : memref<8x8xvector<64xf32>>, memref<4x4xvector<64xf32>, 2>, memref<1xi32>
          affine.dma_wait %alloc_16[%c0_11], %c16_8 : memref<1xi32>
          %6 = affine.apply #map(%arg5, %arg4)
          %7 = affine.apply #map1(%arg5, %arg4)
          %alloc_17 = memref.alloc() : memref<4x4xvector<64xf32>, 2>
          %alloc_18 = memref.alloc() : memref<1xi32>
          affine.dma_start %arg1[%arg5, %arg4], %alloc_17[%c0_7, %c0_7], %alloc_18[%c0_7], %c16_4, %c8_5, %c4_6 : memref<8x8xvector<64xf32>>, memref<4x4xvector<64xf32>, 2>, memref<1xi32>
          affine.dma_wait %alloc_18[%c0_7], %c16_4 : memref<1xi32>
          affine.for %arg6 = #map2(%arg3) to #map3(%arg3) {
            affine.for %arg7 = #map2(%arg4) to #map3(%arg4) {
              affine.for %arg8 = #map2(%arg5) to #map3(%arg5) {
                %8 = affine.load %alloc_15[-%arg3 + %arg6, -%arg5 + %arg8] : memref<4x4xvector<64xf32>, 2>
                %9 = affine.load %alloc_17[-%arg5 + %arg8, -%arg4 + %arg7] : memref<4x4xvector<64xf32>, 2>
                %10 = affine.load %alloc[-%arg3 + %arg6, -%arg4 + %arg7] : memref<4x4xvector<64xf32>, 2>
                %11 = arith.mulf %8, %9 : vector<64xf32>
                %12 = arith.addf %10, %11 : vector<64xf32>
                affine.store %12, %alloc[-%arg3 + %arg6, -%arg4 + %arg7] : memref<4x4xvector<64xf32>, 2>
              }
            }
          }
          memref.dealloc %alloc_18 : memref<1xi32>
          memref.dealloc %alloc_17 : memref<4x4xvector<64xf32>, 2>
          memref.dealloc %alloc_16 : memref<1xi32>
          memref.dealloc %alloc_15 : memref<4x4xvector<64xf32>, 2>
        }
        %2 = affine.apply #map(%arg3, %arg4)
        %3 = affine.apply #map1(%arg3, %arg4)
        affine.dma_start %alloc[%c0, %c0], %arg2[%arg3, %arg4], %alloc_14[%c0], %c16, %c8, %c4 : memref<4x4xvector<64xf32>, 2>, memref<8x8xvector<64xf32>>, memref<1xi32>
        affine.dma_wait %alloc_14[%c0], %c16 : memref<1xi32>
        memref.dealloc %alloc_14 : memref<1xi32>
        memref.dealloc %alloc_13 : memref<1xi32>
        memref.dealloc %alloc : memref<4x4xvector<64xf32>, 2>
      }
    }
    return %arg2 : memref<8x8xvector<64xf32>>
  }
}

