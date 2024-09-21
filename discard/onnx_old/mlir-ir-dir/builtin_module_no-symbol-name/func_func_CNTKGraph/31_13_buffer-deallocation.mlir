// -----// IR Dump After BufferDeallocation (buffer-deallocation) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %alloc = memref.alloc() : memref<1x1x1x1xf32, 1>
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %alloc_2 = memref.alloc() : memref<1x1x1x1xf32, 1>
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %c0_5 = arith.constant 0 : index
  %c0_6 = arith.constant 0 : index
  %c0_7 = arith.constant 0 : index
  %c0_8 = arith.constant 0 : index
  %alloc_9 = memref.alloc() : memref<1x1x1x1xf32, 1>
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
  memref.dealloc %alloc_13 : memref<1x1x28x28xf32>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  affine.parallel (%arg1) = (0) to (1) {
    %8 = affine.apply affine_map<() -> (14)>()
    %9 = affine.apply affine_map<() -> (14)>()
    %10 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%8)[%9]
    %11 = affine.apply affine_map<() -> (8)>()
    %12 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%10)[%11]
    affine.for %arg2 = 0 to %12 {
      %13 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%11]
      %14 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%11]
      %15 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%14)[%9]
      %16 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%14)[%9]
      affine.store %cst_12, %alloc_16[%c0_6, %16, %15, %13] : memref<1x14x14x8xf32>
      %17 = affine.apply affine_map<() -> (2)>()
      %18 = affine.apply affine_map<() -> (2)>()
      %19 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%17)[%18]
      affine.for %arg3 = 0 to %19 {
        %20 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%18]
        %21 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%18]
        %22 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%16, %21)
        %23 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%15, %20)
        affine.store %cst, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
        %24 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%16, %21)
        %25 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%15, %20)
        affine.for %arg4 = 0 to 5 {
          %36 = affine.load %alloc_14[%c0_5, %24 + %arg4, %25 + %c0, %c0_4] : memref<1x32x32x1xf32>
          %37 = affine.load %4[%13, %arg4, %c0, %c0_4] : memref<8x5x5x1xf32>
          %38 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %39 = arith.mulf %36, %37 : f32
          %40 = arith.addf %38, %39 : f32
          affine.store %40, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %41 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c0)
          %42 = affine.load %alloc_14[%c0_5, %24 + %arg4, %25 + %41, %c0_4] : memref<1x32x32x1xf32>
          %43 = affine.load %4[%13, %arg4, %41, %c0_4] : memref<8x5x5x1xf32>
          %44 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %47 = affine.apply affine_map<(d0) -> (d0 + 2)>(%c0)
          %48 = affine.load %alloc_14[%c0_5, %24 + %arg4, %25 + %47, %c0_4] : memref<1x32x32x1xf32>
          %49 = affine.load %4[%13, %arg4, %47, %c0_4] : memref<8x5x5x1xf32>
          %50 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %51 = arith.mulf %48, %49 : f32
          %52 = arith.addf %50, %51 : f32
          affine.store %52, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %53 = affine.apply affine_map<(d0) -> (d0 + 3)>(%c0)
          %54 = affine.load %alloc_14[%c0_5, %24 + %arg4, %25 + %53, %c0_4] : memref<1x32x32x1xf32>
          %55 = affine.load %4[%13, %arg4, %53, %c0_4] : memref<8x5x5x1xf32>
          %56 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %57 = arith.mulf %54, %55 : f32
          %58 = arith.addf %56, %57 : f32
          affine.store %58, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %59 = affine.load %alloc_14[%c0_5, %24 + %arg4, %25 + %c4, %c0_4] : memref<1x32x32x1xf32>
          %60 = affine.load %4[%13, %arg4, %c4, %c0_4] : memref<8x5x5x1xf32>
          %61 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
          %62 = arith.mulf %59, %60 : f32
          %63 = arith.addf %61, %62 : f32
          affine.store %63, %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
        }
        %26 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (0)>(%13, %16, %21, %15, %20)
        %27 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2)>(%13, %16, %21, %15, %20)
        %28 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d3 * 2 + d4)>(%13, %16, %21, %15, %20)
        %29 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)>(%13, %16, %21, %15, %20)
        %30 = affine.load %alloc_2[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
        %31 = affine.load %3[0, %13, 0, 0] : memref<1x8x1x1xf32>
        %32 = affine.load %alloc_16[%arg1, %16, %15, %13] : memref<1x14x14x8xf32>
        %33 = arith.addf %30, %31 : f32
        %34 = arith.maximumf %33, %cst : f32
        %35 = arith.maximumf %32, %34 : f32
        affine.store %35, %alloc_16[%arg1, %16, %15, %13] : memref<1x14x14x8xf32>
      }
    }
  }
  memref.dealloc %alloc_14 : memref<1x32x32x1xf32>
  memref.dealloc %alloc_2 : memref<1x1x1x1xf32, 1>
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
  memref.dealloc %alloc_16 : memref<1x14x14x8xf32>
  %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  affine.parallel (%arg1) = (0) to (1) {
    %8 = affine.apply affine_map<() -> (16)>()
    %9 = affine.apply affine_map<() -> (4)>()
    %10 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%8)[%9]
    %11 = affine.apply affine_map<() -> (4)>()
    %12 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%10)[%11]
    affine.for %arg2 = 0 to %12 {
      %13 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%11]
      %14 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%11]
      %15 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%14)[%9]
      %16 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%14)[%9]
      affine.store %cst_12, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
      %17 = affine.apply affine_map<() -> (3)>()
      %18 = affine.apply affine_map<() -> (3)>()
      %19 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%17)[%18]
      affine.for %arg3 = 0 to %19 {
        %21 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg3)[%18]
        %22 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg3)[%18]
        %23 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%15, %22)
        %24 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%13, %21)
        affine.store %cst, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
        %25 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%15, %22)
        %26 = affine.apply affine_map<(d0, d1) -> (d0 * 3 + d1)>(%13, %21)
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            affine.for %arg6 = 0 to 8 step 4 {
              %37 = affine.load %alloc_17[%c0_11, %25 + %arg4, %26 + %arg5, %arg6] : memref<1x18x18x8xf32>
              %38 = affine.load %1[%16, %arg4, %arg5, %arg6] : memref<16x5x5x8xf32>
              %39 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %40 = arith.mulf %37, %38 : f32
              %41 = arith.addf %39, %40 : f32
              affine.store %41, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %42 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg6)
              %43 = affine.load %alloc_17[%c0_11, %25 + %arg4, %26 + %arg5, %42] : memref<1x18x18x8xf32>
              %44 = affine.load %1[%16, %arg4, %arg5, %42] : memref<16x5x5x8xf32>
              %45 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %46 = arith.mulf %43, %44 : f32
              %47 = arith.addf %45, %46 : f32
              affine.store %47, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %48 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg6)
              %49 = affine.load %alloc_17[%c0_11, %25 + %arg4, %26 + %arg5, %48] : memref<1x18x18x8xf32>
              %50 = affine.load %1[%16, %arg4, %arg5, %48] : memref<16x5x5x8xf32>
              %51 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %52 = arith.mulf %49, %50 : f32
              %53 = arith.addf %51, %52 : f32
              affine.store %53, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %54 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg6)
              %55 = affine.load %alloc_17[%c0_11, %25 + %arg4, %26 + %arg5, %54] : memref<1x18x18x8xf32>
              %56 = affine.load %1[%16, %arg4, %arg5, %54] : memref<16x5x5x8xf32>
              %57 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
              %58 = arith.mulf %55, %56 : f32
              %59 = arith.addf %57, %58 : f32
              affine.store %59, %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
            }
          }
        }
        %27 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (0)>(%16, %15, %22, %13, %21)
        %28 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1 * 3 + d2)>(%16, %15, %22, %13, %21)
        %29 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d3 * 3 + d4)>(%16, %15, %22, %13, %21)
        %30 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)>(%16, %15, %22, %13, %21)
        %31 = affine.load %alloc_9[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
        %32 = affine.load %2[0, %16, 0, 0] : memref<1x16x1x1xf32>
        %33 = affine.load %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
        %34 = arith.addf %31, %32 : f32
        %35 = arith.maximumf %34, %cst : f32
        %36 = arith.maximumf %33, %35 : f32
        affine.store %36, %alloc[0, 0, 0, 0] : memref<1x1x1x1xf32, 1>
      }
      %20 = affine.load %alloc[%arg1, 0, 0, 0] : memref<1x1x1x1xf32, 1>
      affine.store %20, %alloc_19[%arg1, %16, %15, %13] : memref<1x16x4x4xf32>
    }
  }
  memref.dealloc %alloc_17 : memref<1x18x18x8xf32>
  memref.dealloc %alloc_9 : memref<1x1x1x1xf32, 1>
  memref.dealloc %alloc : memref<1x1x1x1xf32, 1>
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
          %16 = affine.load %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %17 = arith.mulf %14, %15 : f32
          %18 = arith.addf %16, %17 : f32
          affine.store %18, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %19 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg4)
          %20 = affine.load %reinterpret_cast_20[%arg1, %arg2, %19] : memref<1x1x256xf32>
          %21 = affine.load %5[%arg1, %19, %arg3] : memref<1x256x10xf32>
          %22 = affine.load %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %23 = arith.mulf %20, %21 : f32
          %24 = arith.addf %22, %23 : f32
          affine.store %24, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %25 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg4)
          %26 = affine.load %reinterpret_cast_20[%arg1, %arg2, %25] : memref<1x1x256xf32>
          %27 = affine.load %5[%arg1, %25, %arg3] : memref<1x256x10xf32>
          %28 = affine.load %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
          %29 = arith.mulf %26, %27 : f32
          %30 = arith.addf %28, %29 : f32
          affine.store %30, %alloc_21[%arg1, %arg2, %arg3] : memref<1x1x10xf32>
        }
      }
    }
  }
  memref.dealloc %alloc_19 : memref<1x16x4x4xf32>
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
  memref.dealloc %alloc_21 : memref<1x1x10xf32>
  %7 = bufferization.to_tensor %alloc_23 : memref<1x10xf32>
  memref.dealloc %alloc_23 : memref<1x10xf32>
  return %7 : tensor<1x10xf32>
}
