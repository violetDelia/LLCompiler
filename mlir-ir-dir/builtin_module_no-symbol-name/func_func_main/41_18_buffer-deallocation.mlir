// -----// IR Dump After BufferDeallocation (buffer-deallocation) //----- //
func.func @main(%arg0: memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x10xf32> {
  %c16 = arith.constant 16 : index
  %c10 = arith.constant 10 : index
  %c-3 = arith.constant -3 : index
  %c-12 = arith.constant -12 : index
  %c3 = arith.constant 3 : index
  %c200 = arith.constant 200 : index
  %c9 = arith.constant 9 : index
  %c256 = arith.constant 256 : index
  %c18 = arith.constant 18 : index
  %c2592 = arith.constant 2592 : index
  %c-5 = arith.constant -5 : index
  %c-2 = arith.constant -2 : index
  %c-28 = arith.constant -28 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c25 = arith.constant 25 : index
  %c4 = arith.constant 4 : index
  %c14 = arith.constant 14 : index
  %c8 = arith.constant 8 : index
  %c1568 = arith.constant 1568 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
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
  %reinterpret_cast = memref.reinterpret_cast %alloc_3 to offset: [0], sizes: [1, 28, 28, 1], strides: [784, 28, 1, 1] : memref<1x1x28x28xf32> to memref<1x28x28x1xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32>
  scf.parallel (%arg1) = (%c0) to (%c1024) step (%c1) {
    %6 = arith.remui %arg1, %c32 : index
    %7 = arith.divui %arg1, %c32 : index
    memref.store %cst_0, %alloc_4[%c0, %7, %6, %c0] : memref<1x32x32x1xf32>
    scf.reduce 
  }
  %reinterpret_cast_5 = memref.reinterpret_cast %alloc_4 to offset: [66], sizes: [1, 28, 28, 1], strides: [1024, 32, 1, 1] : memref<1x32x32x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.copy %reinterpret_cast, %reinterpret_cast_5 : memref<1x28x28x1xf32> to memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  memref.dealloc %alloc_3 : memref<1x1x28x28xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x14x14x8xf32>
  scf.for %arg1 = %c0 to %c1568 step %c1 {
    %6 = arith.remui %arg1, %c8 : index
    %7 = arith.divui %arg1, %c8 : index
    %8 = arith.remui %7, %c14 : index
    %9 = arith.divui %7, %c14 : index
    memref.store %cst, %alloc_6[%c0, %9, %8, %6] : memref<1x14x14x8xf32>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      memref.store %cst_0, %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      scf.for %arg3 = %c0 to %c25 step %c1 {
        %20 = arith.remui %arg3, %c5 : index
        %21 = arith.divui %arg3, %c5 : index
        %22 = arith.muli %9, %c2 : index
        %23 = arith.divui %arg2, %c2 : index
        %24 = arith.addi %22, %23 : index
        %25 = arith.addi %24, %21 : index
        %26 = arith.addi %arg3, %arg2 : index
        %27 = arith.muli %7, %c2 : index
        %28 = arith.addi %26, %27 : index
        %29 = arith.muli %9, %c-28 : index
        %30 = arith.addi %28, %29 : index
        %31 = arith.muli %23, %c-2 : index
        %32 = arith.addi %30, %31 : index
        %33 = arith.muli %21, %c-5 : index
        %34 = arith.addi %32, %33 : index
        %35 = memref.load %alloc_4[%c0, %25, %34, %c0] : memref<1x32x32x1xf32>
        %36 = memref.load %3[%6, %21, %20, %c0] : memref<8x5x5x1xf32>
        %37 = memref.load %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
        %38 = arith.mulf %35, %36 : f32
        %39 = arith.addf %37, %38 : f32
        memref.store %39, %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      }
      %10 = memref.load %alloc_2[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %11 = memref.load %2[%c0, %6, %c0, %c0] : memref<1x8x1x1xf32>
      %12 = memref.load %alloc_6[%c0, %9, %8, %6] : memref<1x14x14x8xf32>
      %13 = arith.addf %10, %11 : f32
      %14 = arith.cmpf ugt, %13, %cst_0 : f32
      %15 = arith.select %14, %13, %cst_0 : f32
      %16 = arith.cmpf ugt, %12, %15 : f32
      %17 = arith.select %16, %12, %15 : f32
      %18 = arith.cmpf uno, %15, %15 : f32
      %19 = arith.select %18, %15, %17 : f32
      memref.store %19, %alloc_6[%c0, %9, %8, %6] : memref<1x14x14x8xf32>
    }
  }
  memref.dealloc %alloc_4 : memref<1x32x32x1xf32>
  memref.dealloc %alloc_2 : memref<1x1x1x1xf32>
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x18x18x8xf32>
  scf.parallel (%arg1) = (%c0) to (%c2592) step (%c1) {
    %6 = arith.remui %arg1, %c8 : index
    %7 = arith.divui %arg1, %c8 : index
    %8 = arith.remui %7, %c18 : index
    %9 = arith.divui %7, %c18 : index
    memref.store %cst_0, %alloc_7[%c0, %9, %8, %6] : memref<1x18x18x8xf32>
    scf.reduce 
  }
  %reinterpret_cast_8 = memref.reinterpret_cast %alloc_7 to offset: [304], sizes: [1, 14, 14, 8], strides: [2592, 144, 8, 1] : memref<1x18x18x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.copy %alloc_6, %reinterpret_cast_8 : memref<1x14x14x8xf32> to memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  memref.dealloc %alloc_6 : memref<1x14x14x8xf32>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x16x4x4xf32>
  scf.for %arg1 = %c0 to %c256 step %c1 {
    %6 = arith.remui %arg1, %c4 : index
    %7 = arith.divui %arg1, %c4 : index
    %8 = arith.remui %7, %c4 : index
    %9 = arith.divui %7, %c4 : index
    memref.store %cst, %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    scf.for %arg2 = %c0 to %c9 step %c1 {
      memref.store %cst_0, %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      scf.for %arg3 = %c0 to %c200 step %c1 {
        %21 = arith.remui %arg3, %c8 : index
        %22 = arith.divui %arg3, %c8 : index
        %23 = arith.remui %22, %c5 : index
        %24 = arith.divui %22, %c5 : index
        %25 = arith.muli %7, %c3 : index
        %26 = arith.muli %9, %c-12 : index
        %27 = arith.addi %25, %26 : index
        %28 = arith.divui %arg2, %c3 : index
        %29 = arith.addi %27, %28 : index
        %30 = arith.addi %29, %24 : index
        %31 = arith.muli %arg1, %c3 : index
        %32 = arith.addi %31, %arg2 : index
        %33 = arith.muli %7, %c-12 : index
        %34 = arith.addi %32, %33 : index
        %35 = arith.muli %28, %c-3 : index
        %36 = arith.addi %34, %35 : index
        %37 = arith.addi %36, %22 : index
        %38 = arith.muli %24, %c-5 : index
        %39 = arith.addi %37, %38 : index
        %40 = memref.load %alloc_7[%c0, %30, %39, %21] : memref<1x18x18x8xf32>
        %41 = memref.load %0[%9, %24, %23, %21] : memref<16x5x5x8xf32>
        %42 = memref.load %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
        %43 = arith.mulf %40, %41 : f32
        %44 = arith.addf %42, %43 : f32
        memref.store %44, %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      }
      %11 = memref.load %alloc[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %12 = memref.load %1[%c0, %9, %c0, %c0] : memref<1x16x1x1xf32>
      %13 = memref.load %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
      %14 = arith.addf %11, %12 : f32
      %15 = arith.cmpf ugt, %14, %cst_0 : f32
      %16 = arith.select %15, %14, %cst_0 : f32
      %17 = arith.cmpf ugt, %13, %16 : f32
      %18 = arith.select %17, %13, %16 : f32
      %19 = arith.cmpf uno, %16, %16 : f32
      %20 = arith.select %19, %16, %18 : f32
      memref.store %20, %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    }
    %10 = memref.load %alloc_1[%c0, %c0, %c0, %c0] : memref<1x1x1x1xf32>
    memref.store %10, %alloc_9[%c0, %9, %8, %6] : memref<1x16x4x4xf32>
  }
  memref.dealloc %alloc_7 : memref<1x18x18x8xf32>
  memref.dealloc %alloc_1 : memref<1x1x1x1xf32>
  memref.dealloc %alloc : memref<1x1x1x1xf32>
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x1x10xf32>
  scf.parallel (%arg1) = (%c0) to (%c10) step (%c1) {
    memref.store %cst_0, %alloc_10[%c0, %c0, %arg1] : memref<1x1x10xf32>
    scf.for %arg2 = %c0 to %c256 step %c1 {
      %6 = arith.divui %arg2, %c16 : index
      %7 = arith.remui %arg2, %c16 : index
      %8 = arith.divui %7, %c4 : index
      %9 = arith.remui %arg2, %c4 : index
      %10 = memref.load %alloc_9[%c0, %6, %8, %9] : memref<1x16x4x4xf32>
      %11 = memref.load %4[%c0, %arg2, %arg1] : memref<1x256x10xf32>
      %12 = memref.load %alloc_10[%c0, %c0, %arg1] : memref<1x1x10xf32>
      %13 = arith.mulf %10, %11 : f32
      %14 = arith.addf %12, %13 : f32
      memref.store %14, %alloc_10[%c0, %c0, %arg1] : memref<1x1x10xf32>
    }
    scf.reduce 
  }
  memref.dealloc %alloc_9 : memref<1x16x4x4xf32>
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
  scf.parallel (%arg1) = (%c0) to (%c10) step (%c1) {
    %6 = memref.load %alloc_10[%c0, %c0, %arg1] : memref<1x1x10xf32>
    %7 = memref.load %5[%c0, %arg1] : memref<1x10xf32>
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %alloc_11[%c0, %arg1] : memref<1x10xf32>
    scf.reduce 
  }
  memref.dealloc %alloc_10 : memref<1x1x10xf32>
  return %alloc_11 : memref<1x10xf32>
}

