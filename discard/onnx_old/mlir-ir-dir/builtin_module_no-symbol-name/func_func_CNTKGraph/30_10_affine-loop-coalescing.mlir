// -----// IR Dump After LoopCoalescing (affine-loop-coalescing) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %alloc = memref.alloc() : memref<1x1x1x1xf32>
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %alloc_1 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %c0_5 = arith.constant 0 : index
  %c0_6 = arith.constant 0 : index
  %c0_7 = arith.constant 0 : index
  %alloc_8 = memref.alloc() : memref<1x1x1x1xf32>
  %c0_9 = arith.constant 0 : index
  %c0_10 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_11 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %2 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %3 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %4 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %5 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %6 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %0, %alloc_12 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc_12 to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %7 = affine.apply affine_map<() -> (1)>()
  %8 = affine.apply affine_map<() -> (32)>()
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%7)[%8]
  %10 = affine.apply affine_map<() -> (32)>()
  %11 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%9)[%10]
  %12 = affine.apply affine_map<() -> (1)>()
  %13 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%11)[%12]
  affine.for %arg1 = 0 to %13 {
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%12]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%12]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%10]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%10]
    %48 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%47)[%8]
    %49 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%47)[%8]
    affine.store %cst, %alloc_13[%49, %48, %46, %44] : memref<1x32x32x1xf32>
  }
  %reinterpret_cast_14 = memref.reinterpret_cast %alloc_13 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_14 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %14 = affine.apply affine_map<() -> (1)>()
  %15 = affine.apply affine_map<() -> (14)>()
  %16 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%14)[%15]
  %17 = affine.apply affine_map<() -> (14)>()
  %18 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%16)[%17]
  %19 = affine.apply affine_map<() -> (8)>()
  %20 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%18)[%19]
  affine.for %arg1 = 0 to %20 {
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%19]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%19]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%17]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%17]
    %48 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%47)[%15]
    %49 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%47)[%15]
    affine.store %cst_11, %alloc_15[%c0_5, %48, %46, %44] : memref<1x14x14x8xf32>
    %50 = affine.apply affine_map<() -> (2)>()
    %51 = affine.apply affine_map<() -> (2)>()
    %52 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%50)[%51]
    affine.for %arg2 = 0 to %52 {
      %53 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%51]
      %54 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%51]
      %55 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%48, %54)
      %56 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%46, %53)
      affine.store %cst, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %57 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%48, %54)
      %58 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%46, %53)
      %59 = affine.apply affine_map<() -> (5)>()
      %60 = affine.apply affine_map<() -> (5)>()
      %61 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%59)[%60]
      affine.for %arg3 = 0 to %61 {
        %72 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%60]
        %73 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%60]
        %74 = affine.load %alloc_13[%c0_4, %57 + %73, %58 + %72, %c0_3] : memref<1x32x32x1xf32>
        %75 = affine.load %4[%44, %73, %72, %c0_3] : memref<8x5x5x1xf32>
        %76 = affine.load %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %77 = arith.mulf %74, %75 : f32
        %78 = arith.addf %76, %77 : f32
        affine.store %78, %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %62 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (0)>(%44, %48, %54, %46, %53)
      %63 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2)>(%44, %48, %54, %46, %53)
      %64 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d3 * 2 + d4)>(%44, %48, %54, %46, %53)
      %65 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)>(%44, %48, %54, %46, %53)
      %66 = affine.load %alloc_1[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %67 = affine.load %3[0, %44, 0, 0] : memref<1x8x1x1xf32>
      %68 = affine.load %alloc_15[%49, %48, %46, %44] : memref<1x14x14x8xf32>
      %69 = arith.addf %66, %67 : f32
      %70 = arith.maximumf %69, %cst : f32
      %71 = arith.maximumf %68, %70 : f32
      affine.store %71, %alloc_15[%49, %48, %46, %44] : memref<1x14x14x8xf32>
    }
  }
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %21 = affine.apply affine_map<() -> (1)>()
  %22 = affine.apply affine_map<() -> (18)>()
  %23 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%21)[%22]
  %24 = affine.apply affine_map<() -> (18)>()
  %25 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%23)[%24]
  %26 = affine.apply affine_map<() -> (8)>()
  %27 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%25)[%26]
  affine.for %arg1 = 0 to %27 {
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%26]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%26]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%24]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%24]
    %48 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%47)[%22]
    %49 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%47)[%22]
    affine.store %cst, %alloc_16[%49, %48, %46, %44] : memref<1x18x18x8xf32>
  }
  %reinterpret_cast_17 = memref.reinterpret_cast %alloc_16 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_15, %reinterpret_cast_17 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %28 = affine.apply affine_map<() -> (1)>()
  %29 = affine.apply affine_map<() -> (16)>()
  %30 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%28)[%29]
  %31 = affine.apply affine_map<() -> (4)>()
  %32 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%30)[%31]
  %33 = affine.apply affine_map<() -> (4)>()
  %34 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%32)[%33]
  affine.for %arg1 = 0 to %34 {
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%33]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%33]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%31]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%31]
    %48 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%47)[%29]
    %49 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%47)[%29]
    affine.store %cst_11, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
    %50 = affine.apply affine_map<() -> (3)>()
    %51 = affine.apply affine_map<() -> (3)>()
    %52 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%50)[%51]
    affine.for %arg2 = 0 to %52 {
      %54 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%51]
      %55 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%51]
      %56 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%46, %55)
      %57 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%44, %54)
      affine.store %cst, %alloc_8[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %58 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%46, %55)
      %59 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%44, %54)
      %60 = affine.apply affine_map<() -> (5)>()
      %61 = affine.apply affine_map<() -> (5)>()
      %62 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%60)[%61]
      %63 = affine.apply affine_map<() -> (8)>()
      %64 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%62)[%63]
      affine.for %arg3 = 0 to %64 {
        %75 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%63]
        %76 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%63]
        %77 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%76)[%61]
        %78 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%76)[%61]
        %79 = affine.load %alloc_16[%c0_10, %58 + %78, %59 + %77, %75] : memref<1x18x18x8xf32>
        %80 = affine.load %1[%48, %78, %77, %75] : memref<16x5x5x8xf32>
        %81 = affine.load %alloc_8[0, 0, 0, 0] : memref<1x1x1x1xf32>
        %82 = arith.mulf %79, %80 : f32
        %83 = arith.addf %81, %82 : f32
        affine.store %83, %alloc_8[0, 0, 0, 0] : memref<1x1x1x1xf32>
      }
      %65 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (0)>(%48, %46, %55, %44, %54)
      %66 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1 * 3 + d2)>(%48, %46, %55, %44, %54)
      %67 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d3 * 3 + d4)>(%48, %46, %55, %44, %54)
      %68 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)>(%48, %46, %55, %44, %54)
      %69 = affine.load %alloc_8[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %70 = affine.load %2[0, %48, 0, 0] : memref<1x16x1x1xf32>
      %71 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
      %72 = arith.addf %69, %70 : f32
      %73 = arith.maximumf %72, %cst : f32
      %74 = arith.maximumf %71, %73 : f32
      affine.store %74, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32>
    }
    %53 = affine.load %alloc[%49, 0, 0, 0] : memref<1x1x1x1xf32>
    affine.store %53, %alloc_18[%49, %48, %46, %44] : memref<1x16x4x4xf32>
  }
  %reinterpret_cast_19 = memref.reinterpret_cast %alloc_18 to offset: [0], sizes: [1, 1, 256], strides: [256, 256, 1] : memref<1x16x4x4xf32> to memref<1x1x256xf32>
  %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %35 = affine.apply affine_map<() -> (1)>()
  %36 = affine.apply affine_map<() -> (1)>()
  %37 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%35)[%36]
  %38 = affine.apply affine_map<() -> (10)>()
  %39 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%37)[%38]
  affine.for %arg1 = 0 to %39 {
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%38]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%38]
    %46 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%45)[%36]
    %47 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%45)[%36]
    affine.store %cst, %alloc_20[%c0_7, %c0_6, %44] : memref<1x1x10xf32>
    affine.for %arg2 = 0 to 256 {
      %48 = affine.load %reinterpret_cast_19[%47, %46, %arg2] : memref<1x1x256xf32>
      %49 = affine.load %5[%47, %arg2, %44] : memref<1x256x10xf32>
      %50 = affine.load %alloc_20[%47, %46, %44] : memref<1x1x10xf32>
      %51 = arith.mulf %48, %49 : f32
      %52 = arith.addf %50, %51 : f32
      affine.store %52, %alloc_20[%47, %46, %44] : memref<1x1x10xf32>
    }
  }
  %reinterpret_cast_21 = memref.reinterpret_cast %alloc_20 to offset: [0], sizes: [1, 10], strides: [10, 1] : memref<1x1x10xf32> to memref<1x10xf32>
  %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %40 = affine.apply affine_map<() -> (1)>()
  %41 = affine.apply affine_map<() -> (10)>()
  %42 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%40)[%41]
  affine.for %arg1 = 0 to %42 {
    %44 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%41]
    %45 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%41]
    %46 = affine.load %reinterpret_cast_21[0, %44] : memref<1x10xf32>
    %47 = affine.load %6[0, %44] : memref<1x10xf32>
    %48 = arith.addf %46, %47 : f32
    affine.store %48, %alloc_22[%45, %44] : memref<1x10xf32>
  }
  %43 = bufferization.to_tensor %alloc_22 : memref<1x10xf32>
  return %43 : tensor<1x10xf32>
}

