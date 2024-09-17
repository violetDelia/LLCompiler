// -----// IR Dump After LoopCoalescing (affine-loop-coalescing) //----- //
func.func @CNTKGraph(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_5 = arith.constant -3.40282347E+38 : f32
  %0 = bufferization.to_memref %arg0 : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %1 = memref.get_global @__constant_16x5x5x8xf32 : memref<16x5x5x8xf32>
  %2 = memref.get_global @__constant_1x16x1x1xf32 : memref<1x16x1x1xf32>
  %3 = memref.get_global @__constant_1x8x1x1xf32 : memref<1x8x1x1xf32>
  %4 = memref.get_global @__constant_8x5x5x1xf32 : memref<8x5x5x1xf32>
  %5 = memref.get_global @__constant_1x256x10xf32 : memref<1x256x10xf32>
  %6 = memref.get_global @__constant_1x10xf32 : memref<1x10xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x28x28xf32>
  memref.copy %0, %alloc : memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x28x28xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  %7 = affine.apply affine_map<() -> (1)>()
  %8 = affine.apply affine_map<() -> (32)>()
  %9 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%7)[%8]
  %10 = affine.apply affine_map<() -> (32)>()
  %11 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%9)[%10]
  %12 = affine.apply affine_map<() -> (1)>()
  %13 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%11)[%12]
  affine.for %arg1 = 0 to %13 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%12]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%12]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%10]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%10]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%8]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%8]
    affine.store %cst, %alloc_6[%79, %78, %76, %74] : memref<1x32x32x1xf32>
  }
  %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_7 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc : memref<1x1x28x28xf32>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x28x28x8xf32>
  %14 = affine.apply affine_map<() -> (1)>()
  %15 = affine.apply affine_map<() -> (28)>()
  %16 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%14)[%15]
  %17 = affine.apply affine_map<() -> (28)>()
  %18 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%16)[%17]
  %19 = affine.apply affine_map<() -> (8)>()
  %20 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%18)[%19]
  affine.for %arg1 = 0 to %20 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%19]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%19]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%17]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%17]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%15]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%15]
    affine.store %cst, %alloc_8[%79, %78, %76, %74] : memref<1x28x28x8xf32>
  }
  %21 = affine.apply affine_map<() -> (1)>()
  %22 = affine.apply affine_map<() -> (28)>()
  %23 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%21)[%22]
  %24 = affine.apply affine_map<() -> (28)>()
  %25 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%23)[%24]
  %26 = affine.apply affine_map<() -> (8)>()
  %27 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%25)[%26]
  %28 = affine.apply affine_map<() -> (1)>()
  %29 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%27)[%28]
  affine.for %arg1 = 0 to %29 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%28]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%28]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%26]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%26]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%24]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%24]
    %80 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%79)[%22]
    %81 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%79)[%22]
    affine.store %cst, %alloc_8[%c0_3, %80, %78, %76] : memref<1x28x28x8xf32>
    %82 = affine.apply affine_map<() -> (5)>()
    %83 = affine.apply affine_map<() -> (5)>()
    %84 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%82)[%83]
    affine.for %arg2 = 0 to %84 {
      %85 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%83]
      %86 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%83]
      %87 = affine.load %alloc_6[%81, %80 + %86, %78 + %85, %74] : memref<1x32x32x1xf32>
      %88 = affine.load %4[%76, %86, %85, %74] : memref<8x5x5x1xf32>
      %89 = affine.load %alloc_8[%81, %80, %78, %76] : memref<1x28x28x8xf32>
      %90 = arith.mulf %87, %88 : f32
      %91 = arith.addf %89, %90 : f32
      affine.store %91, %alloc_8[%81, %80, %78, %76] : memref<1x28x28x8xf32>
    }
  }
  memref.dealloc %alloc_6 : memref<1x32x32x1xf32>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  %30 = affine.apply affine_map<() -> (1)>()
  %31 = affine.apply affine_map<() -> (14)>()
  %32 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%30)[%31]
  %33 = affine.apply affine_map<() -> (14)>()
  %34 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%32)[%33]
  %35 = affine.apply affine_map<() -> (8)>()
  %36 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%34)[%35]
  affine.for %arg1 = 0 to %36 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%35]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%35]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%33]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%33]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%31]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%31]
    affine.store %cst_5, %alloc_9[%c0_4, %78, %76, %74] : memref<1x14x14x8xf32>
    %80 = affine.apply affine_map<() -> (2)>()
    %81 = affine.apply affine_map<() -> (2)>()
    %82 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%80)[%81]
    affine.for %arg2 = 0 to %82 {
      %83 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%81]
      %84 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%81]
      %85 = affine.load %alloc_8[0, %78 * 2 + %84, %76 * 2 + %83, %74] : memref<1x28x28x8xf32>
      %86 = affine.load %3[0, %74, 0, 0] : memref<1x8x1x1xf32>
      %87 = affine.load %alloc_9[%79, %78, %76, %74] : memref<1x14x14x8xf32>
      %88 = arith.addf %85, %86 : f32
      %89 = arith.maximumf %88, %cst : f32
      %90 = arith.maximumf %87, %89 : f32
      affine.store %90, %alloc_9[%79, %78, %76, %74] : memref<1x14x14x8xf32>
    }
  }
  memref.dealloc %alloc_8 : memref<1x28x28x8xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  %37 = affine.apply affine_map<() -> (1)>()
  %38 = affine.apply affine_map<() -> (18)>()
  %39 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%37)[%38]
  %40 = affine.apply affine_map<() -> (18)>()
  %41 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%39)[%40]
  %42 = affine.apply affine_map<() -> (8)>()
  %43 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%41)[%42]
  affine.for %arg1 = 0 to %43 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%42]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%42]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%40]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%40]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%38]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%38]
    affine.store %cst, %alloc_10[%79, %78, %76, %74] : memref<1x18x18x8xf32>
  }
  %reinterpret_cast_11 = memref.reinterpret_cast %alloc_10 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_9, %reinterpret_cast_11 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_9 : memref<1x14x14x8xf32>
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x16xf32>
  %44 = affine.apply affine_map<() -> (1)>()
  %45 = affine.apply affine_map<() -> (14)>()
  %46 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%44)[%45]
  %47 = affine.apply affine_map<() -> (14)>()
  %48 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%46)[%47]
  %49 = affine.apply affine_map<() -> (16)>()
  %50 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%48)[%49]
  affine.for %arg1 = 0 to %50 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%49]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%49]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%47]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%47]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%45]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%45]
    affine.store %cst, %alloc_12[%c0, %78, %76, %74] : memref<1x14x14x16xf32>
    %80 = affine.apply affine_map<() -> (5)>()
    %81 = affine.apply affine_map<() -> (5)>()
    %82 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%80)[%81]
    %83 = affine.apply affine_map<() -> (8)>()
    %84 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%82)[%83]
    affine.for %arg2 = 0 to %84 {
      %85 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%83]
      %86 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%83]
      %87 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%86)[%81]
      %88 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%86)[%81]
      %89 = affine.load %alloc_10[%79, %78 + %88, %76 + %87, %85] : memref<1x18x18x8xf32>
      %90 = affine.load %1[%74, %88, %87, %85] : memref<16x5x5x8xf32>
      %91 = affine.load %alloc_12[%79, %78, %76, %74] : memref<1x14x14x16xf32>
      %92 = arith.mulf %89, %90 : f32
      %93 = arith.addf %91, %92 : f32
      affine.store %93, %alloc_12[%79, %78, %76, %74] : memref<1x14x14x16xf32>
    }
  }
  memref.dealloc %alloc_10 : memref<1x18x18x8xf32>
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4x16xf32>
  %51 = affine.apply affine_map<() -> (1)>()
  %52 = affine.apply affine_map<() -> (4)>()
  %53 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%51)[%52]
  %54 = affine.apply affine_map<() -> (4)>()
  %55 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%53)[%54]
  %56 = affine.apply affine_map<() -> (16)>()
  %57 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%55)[%56]
  affine.for %arg1 = 0 to %57 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%56]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%56]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%54]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%54]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%52]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%52]
    affine.store %cst_5, %alloc_13[%c0_0, %78, %76, %74] : memref<1x4x4x16xf32>
    %80 = affine.apply affine_map<() -> (3)>()
    %81 = affine.apply affine_map<() -> (3)>()
    %82 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%80)[%81]
    affine.for %arg2 = 0 to %82 {
      %83 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg2)[%81]
      %84 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg2)[%81]
      %85 = affine.load %alloc_12[0, %78 * 3 + %84, %76 * 3 + %83, %74] : memref<1x14x14x16xf32>
      %86 = affine.load %2[0, %74, 0, 0] : memref<1x16x1x1xf32>
      %87 = affine.load %alloc_13[%79, %78, %76, %74] : memref<1x4x4x16xf32>
      %88 = arith.addf %85, %86 : f32
      %89 = arith.maximumf %88, %cst : f32
      %90 = arith.maximumf %87, %89 : f32
      affine.store %90, %alloc_13[%79, %78, %76, %74] : memref<1x4x4x16xf32>
    }
  }
  memref.dealloc %alloc_12 : memref<1x14x14x16xf32>
  %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  %58 = affine.apply affine_map<() -> (1)>()
  %59 = affine.apply affine_map<() -> (16)>()
  %60 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%58)[%59]
  %61 = affine.apply affine_map<() -> (4)>()
  %62 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%60)[%61]
  %63 = affine.apply affine_map<() -> (4)>()
  %64 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%62)[%63]
  affine.for %arg1 = 0 to %64 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%63]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%63]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%61]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%61]
    %78 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%77)[%59]
    %79 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%77)[%59]
    %80 = affine.load %alloc_13[%79, %76, %74, %78] : memref<1x4x4x16xf32>
    affine.store %80, %alloc_14[%79, %78, %76, %74] : memref<1x16x4x4xf32>
  }
  memref.dealloc %alloc_13 : memref<1x4x4x16xf32>
  %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  %65 = affine.apply affine_map<() -> (1)>()
  %66 = affine.apply affine_map<() -> (1)>()
  %67 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%65)[%66]
  %68 = affine.apply affine_map<() -> (10)>()
  %69 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%67)[%68]
  affine.for %arg1 = 0 to %69 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%68]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%68]
    %76 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%75)[%66]
    %77 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%75)[%66]
    affine.store %cst, %alloc_15[%c0_2, %c0_1, %74] : memref<1x1x10xf32>
    affine.for %arg2 = 0 to 256 {
      %78 = affine.load %alloc_14[symbol(%77) + symbol(%76), %arg2 floordiv 16, (%arg2 mod 16) floordiv 4, %arg2 mod 4] : memref<1x16x4x4xf32>
      %79 = affine.load %5[%77, %arg2, %74] : memref<1x256x10xf32>
      %80 = affine.load %alloc_15[%77, %76, %74] : memref<1x1x10xf32>
      %81 = arith.mulf %78, %79 : f32
      %82 = arith.addf %80, %81 : f32
      affine.store %82, %alloc_15[%77, %76, %74] : memref<1x1x10xf32>
    }
  }
  memref.dealloc %alloc_14 : memref<1x16x4x4xf32>
  %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  %70 = affine.apply affine_map<() -> (1)>()
  %71 = affine.apply affine_map<() -> (10)>()
  %72 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%70)[%71]
  affine.for %arg1 = 0 to %72 {
    %74 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%arg1)[%71]
    %75 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%arg1)[%71]
    %76 = affine.load %alloc_15[0, 0, %74] : memref<1x1x10xf32>
    %77 = affine.load %6[0, %74] : memref<1x10xf32>
    %78 = arith.addf %76, %77 : f32
    affine.store %78, %alloc_16[%75, %74] : memref<1x10xf32>
  }
  memref.dealloc %alloc_15 : memref<1x1x10xf32>
  %73 = bufferization.to_tensor %alloc_16 : memref<1x10xf32>
  memref.dealloc %alloc_16 : memref<1x10xf32>
  return %73 : tensor<1x10xf32>
}

