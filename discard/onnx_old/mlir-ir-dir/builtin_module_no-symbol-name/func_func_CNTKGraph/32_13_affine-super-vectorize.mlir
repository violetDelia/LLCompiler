// -----// IR Dump After AffineVectorize Failed (affine-super-vectorize) //----- //
"func.func"() <{function_type = (tensor<1x1x28x28xf32>) -> tensor<1x10xf32>, sym_name = "CNTKGraph"}> ({
^bb0(%arg0: tensor<1x1x28x28xf32>):
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %1 = "arith.constant"() <{value = 0 : index}> : () -> index
  %2 = "arith.constant"() <{value = 0 : index}> : () -> index
  %3 = "arith.constant"() <{value = 0 : index}> : () -> index
  %4 = "arith.constant"() <{value = 0 : index}> : () -> index
  %5 = "arith.constant"() <{value = 0 : index}> : () -> index
  %6 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
  %7 = "arith.constant"() <{value = -3.40282347E+38 : f32}> : () -> f32
  %8 = "bufferization.to_memref"(%arg0) : (tensor<1x1x28x28xf32>) -> memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>
  %9 = "memref.get_global"() <{name = @__constant_16x5x5x8xf32}> : () -> memref<16x5x5x8xf32>
  %10 = "memref.get_global"() <{name = @__constant_1x16x1x1xf32}> : () -> memref<1x16x1x1xf32>
  %11 = "memref.get_global"() <{name = @__constant_1x8x1x1xf32}> : () -> memref<1x8x1x1xf32>
  %12 = "memref.get_global"() <{name = @__constant_8x5x5x1xf32}> : () -> memref<8x5x5x1xf32>
  %13 = "memref.get_global"() <{name = @__constant_1x256x10xf32}> : () -> memref<1x256x10xf32>
  %14 = "memref.get_global"() <{name = @__constant_1x10xf32}> : () -> memref<1x10xf32>
  %15 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x28x28xf32>
  "memref.copy"(%8, %15) : (memref<1x1x28x28xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<1x1x28x28xf32>) -> ()
  %16 = "memref.reinterpret_cast"(%15) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0>, static_sizes = array<i64: 1, 28, 28, 1>, static_strides = array<i64: 784, 28, 1, 1>}> : (memref<1x1x28x28xf32>) -> memref<1x28x28x1xf32>
  %17 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x32x32x1xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg45: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
    ^bb0(%arg46: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (32)>}> ({
      ^bb0(%arg47: index):
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (1)>}> ({
        ^bb0(%arg48: index):
          %120 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
          "vector.transfer_write"(%120, %17, %arg45, %arg46, %arg47, %arg48) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>}> : (vector<1024xf32>, memref<1x32x32x1xf32>, index, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  %18 = "memref.reinterpret_cast"(%17) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 66>, static_sizes = array<i64: 1, 28, 28, 1>, static_strides = array<i64: 1024, 32, 1, 1>}> : (memref<1x32x32x1xf32>) -> memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>
  "memref.copy"(%16, %18) : (memref<1x28x28x1xf32>, memref<1x28x28x1xf32, strided<[1024, 32, 1, 1], offset: 66>>) -> ()
  "memref.dealloc"(%15) : (memref<1x1x28x28xf32>) -> ()
  %19 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x28x28x8xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg41: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (28)>}> ({
    ^bb0(%arg42: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (28)>}> ({
      ^bb0(%arg43: index):
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (8)>}> ({
        ^bb0(%arg44: index):
          %119 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
          "vector.transfer_write"(%119, %19, %arg41, %arg42, %arg43, %arg44) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>}> : (vector<1024xf32>, memref<1x28x28x8xf32>, index, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg34: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (28)>}> ({
    ^bb0(%arg35: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (28)>}> ({
      ^bb0(%arg36: index):
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (8)>}> ({
        ^bb0(%arg37: index):
          "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (1)>}> ({
          ^bb0(%arg38: index):
            %106 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
            "vector.transfer_write"(%106, %19, %4, %arg35, %arg36, %arg37) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (vector<1024xf32>, memref<1x28x28x8xf32>, index, index, index, index) -> ()
            "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (5)>}> ({
            ^bb0(%arg39: index):
              "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (5)>}> ({
              ^bb0(%arg40: index):
                %107 = "affine.apply"(%arg34, %arg38, %arg35, %arg39, %arg36, %arg40) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0)>}> : (index, index, index, index, index, index) -> index
                %108 = "affine.apply"(%arg34, %arg38, %arg35, %arg39, %arg36, %arg40) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2 + d3)>}> : (index, index, index, index, index, index) -> index
                %109 = "affine.apply"(%arg34, %arg38, %arg35, %arg39, %arg36, %arg40) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4 + d5)>}> : (index, index, index, index, index, index) -> index
                %110 = "affine.apply"(%arg34, %arg38, %arg35, %arg39, %arg36, %arg40) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1)>}> : (index, index, index, index, index, index) -> index
                %111 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
                %112 = "vector.transfer_read"(%17, %107, %108, %109, %110, %111) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>}> : (memref<1x32x32x1xf32>, index, index, index, index, f32) -> vector<1024xf32>
                %113 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
                %114 = "vector.transfer_read"(%12, %arg37, %arg39, %arg40, %arg38, %113) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>}> : (memref<8x5x5x1xf32>, index, index, index, index, f32) -> vector<1024xf32>
                %115 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
                %116 = "vector.transfer_read"(%19, %arg34, %arg35, %arg36, %arg37, %115) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (memref<1x28x28x8xf32>, index, index, index, index, f32) -> vector<1024xf32>
                %117 = "arith.mulf"(%112, %114) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
                %118 = "arith.addf"(%116, %117) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
                "vector.transfer_write"(%118, %19, %arg34, %arg35, %arg36, %arg37) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (vector<1024xf32>, memref<1x28x28x8xf32>, index, index, index, index) -> ()
                "affine.yield"() : () -> ()
              }) : () -> ()
              "affine.yield"() : () -> ()
            }) : () -> ()
            "affine.yield"() : () -> ()
          }) : () -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%17) : (memref<1x32x32x1xf32>) -> ()
  %20 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x14x14x8xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg28: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (14)>}> ({
    ^bb0(%arg29: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (14)>}> ({
      ^bb0(%arg30: index):
        %87 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
        %88 = "arith.constant"() <{value = dense<-3.40282347E+38> : vector<1024xf32>}> : () -> vector<1024xf32>
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (8)>}> ({
        ^bb0(%arg31: index):
          "vector.transfer_write"(%88, %20, %5, %arg29, %arg30, %arg31) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (vector<1024xf32>, memref<1x14x14x8xf32>, index, index, index, index) -> ()
          "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}> ({
          ^bb0(%arg32: index):
            "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (2)>}> ({
            ^bb0(%arg33: index):
              %89 = "affine.apply"(%arg31, %arg29, %arg32, %arg30, %arg33) <{map = affine_map<(d0, d1, d2, d3, d4) -> (0)>}> : (index, index, index, index, index) -> index
              %90 = "affine.apply"(%arg31, %arg29, %arg32, %arg30, %arg33) <{map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2)>}> : (index, index, index, index, index) -> index
              %91 = "affine.apply"(%arg31, %arg29, %arg32, %arg30, %arg33) <{map = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 2 + d4)>}> : (index, index, index, index, index) -> index
              %92 = "affine.apply"(%arg31, %arg29, %arg32, %arg30, %arg33) <{map = affine_map<(d0, d1, d2, d3, d4) -> (d0)>}> : (index, index, index, index, index) -> index
              %93 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
              %94 = "vector.transfer_read"(%19, %89, %90, %91, %92, %93) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (memref<1x28x28x8xf32>, index, index, index, index, f32) -> vector<1024xf32>
              %95 = "affine.apply"(%arg31) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
              %96 = "affine.apply"(%arg31) <{map = affine_map<(d0) -> (d0)>}> : (index) -> index
              %97 = "affine.apply"(%arg31) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
              %98 = "affine.apply"(%arg31) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
              %99 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
              %100 = "vector.transfer_read"(%11, %95, %96, %97, %98, %99) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (memref<1x8x1x1xf32>, index, index, index, index, f32) -> vector<1024xf32>
              %101 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
              %102 = "vector.transfer_read"(%20, %arg28, %arg29, %arg30, %arg31, %101) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (memref<1x14x14x8xf32>, index, index, index, index, f32) -> vector<1024xf32>
              %103 = "arith.addf"(%94, %100) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
              %104 = "arith.maximumf"(%103, %87) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
              %105 = "arith.maximumf"(%102, %104) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
              "vector.transfer_write"(%105, %20, %arg28, %arg29, %arg30, %arg31) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (vector<1024xf32>, memref<1x14x14x8xf32>, index, index, index, index) -> ()
              "affine.yield"() : () -> ()
            }) : () -> ()
            "affine.yield"() : () -> ()
          }) : () -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%19) : (memref<1x28x28x8xf32>) -> ()
  %21 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x18x18x8xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg24: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (18)>}> ({
    ^bb0(%arg25: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (18)>}> ({
      ^bb0(%arg26: index):
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (8)>}> ({
        ^bb0(%arg27: index):
          %86 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
          "vector.transfer_write"(%86, %21, %arg24, %arg25, %arg26, %arg27) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>}> : (vector<1024xf32>, memref<1x18x18x8xf32>, index, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  %22 = "memref.reinterpret_cast"(%21) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 304>, static_sizes = array<i64: 1, 14, 14, 8>, static_strides = array<i64: 2592, 144, 8, 1>}> : (memref<1x18x18x8xf32>) -> memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>
  "memref.copy"(%20, %22) : (memref<1x14x14x8xf32>, memref<1x14x14x8xf32, strided<[2592, 144, 8, 1], offset: 304>>) -> ()
  "memref.dealloc"(%20) : (memref<1x14x14x8xf32>) -> ()
  %23 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x14x14x16xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg17: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (14)>}> ({
    ^bb0(%arg18: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (14)>}> ({
      ^bb0(%arg19: index):
        %73 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (16)>}> ({
        ^bb0(%arg20: index):
          "vector.transfer_write"(%73, %23, %0, %arg18, %arg19, %arg20) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (vector<1024xf32>, memref<1x14x14x16xf32>, index, index, index, index) -> ()
          "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (5)>}> ({
          ^bb0(%arg21: index):
            "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (5)>}> ({
            ^bb0(%arg22: index):
              "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (8)>}> ({
              ^bb0(%arg23: index):
                %74 = "affine.apply"(%arg17, %arg23, %arg18, %arg21, %arg19, %arg22) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0)>}> : (index, index, index, index, index, index) -> index
                %75 = "affine.apply"(%arg17, %arg23, %arg18, %arg21, %arg19, %arg22) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2 + d3)>}> : (index, index, index, index, index, index) -> index
                %76 = "affine.apply"(%arg17, %arg23, %arg18, %arg21, %arg19, %arg22) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4 + d5)>}> : (index, index, index, index, index, index) -> index
                %77 = "affine.apply"(%arg17, %arg23, %arg18, %arg21, %arg19, %arg22) <{map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1)>}> : (index, index, index, index, index, index) -> index
                %78 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
                %79 = "vector.transfer_read"(%21, %74, %75, %76, %77, %78) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (memref<1x18x18x8xf32>, index, index, index, index, f32) -> vector<1024xf32>
                %80 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
                %81 = "vector.transfer_read"(%9, %arg20, %arg21, %arg22, %arg23, %80) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (memref<16x5x5x8xf32>, index, index, index, index, f32) -> vector<1024xf32>
                %82 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
                %83 = "vector.transfer_read"(%23, %arg17, %arg18, %arg19, %arg20, %82) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (memref<1x14x14x16xf32>, index, index, index, index, f32) -> vector<1024xf32>
                %84 = "arith.mulf"(%79, %81) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
                %85 = "arith.addf"(%83, %84) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
                "vector.transfer_write"(%85, %23, %arg17, %arg18, %arg19, %arg20) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (vector<1024xf32>, memref<1x14x14x16xf32>, index, index, index, index) -> ()
                "affine.yield"() : () -> ()
              }) : () -> ()
              "affine.yield"() : () -> ()
            }) : () -> ()
            "affine.yield"() : () -> ()
          }) : () -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%21) : (memref<1x18x18x8xf32>) -> ()
  %24 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x4x4x16xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg11: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
    ^bb0(%arg12: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (4)>}> ({
      ^bb0(%arg13: index):
        %54 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
        %55 = "arith.constant"() <{value = dense<-3.40282347E+38> : vector<1024xf32>}> : () -> vector<1024xf32>
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (16)>}> ({
        ^bb0(%arg14: index):
          "vector.transfer_write"(%55, %24, %1, %arg12, %arg13, %arg14) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (vector<1024xf32>, memref<1x4x4x16xf32>, index, index, index, index) -> ()
          "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (3)>}> ({
          ^bb0(%arg15: index):
            "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (3)>}> ({
            ^bb0(%arg16: index):
              %56 = "affine.apply"(%arg14, %arg12, %arg15, %arg13, %arg16) <{map = affine_map<(d0, d1, d2, d3, d4) -> (0)>}> : (index, index, index, index, index) -> index
              %57 = "affine.apply"(%arg14, %arg12, %arg15, %arg13, %arg16) <{map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 3 + d2)>}> : (index, index, index, index, index) -> index
              %58 = "affine.apply"(%arg14, %arg12, %arg15, %arg13, %arg16) <{map = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 3 + d4)>}> : (index, index, index, index, index) -> index
              %59 = "affine.apply"(%arg14, %arg12, %arg15, %arg13, %arg16) <{map = affine_map<(d0, d1, d2, d3, d4) -> (d0)>}> : (index, index, index, index, index) -> index
              %60 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
              %61 = "vector.transfer_read"(%23, %56, %57, %58, %59, %60) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (memref<1x14x14x16xf32>, index, index, index, index, f32) -> vector<1024xf32>
              %62 = "affine.apply"(%arg14) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
              %63 = "affine.apply"(%arg14) <{map = affine_map<(d0) -> (d0)>}> : (index) -> index
              %64 = "affine.apply"(%arg14) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
              %65 = "affine.apply"(%arg14) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
              %66 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
              %67 = "vector.transfer_read"(%10, %62, %63, %64, %65, %66) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (memref<1x16x1x1xf32>, index, index, index, index, f32) -> vector<1024xf32>
              %68 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
              %69 = "vector.transfer_read"(%24, %arg11, %arg12, %arg13, %arg14, %68) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (memref<1x4x4x16xf32>, index, index, index, index, f32) -> vector<1024xf32>
              %70 = "arith.addf"(%61, %67) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
              %71 = "arith.maximumf"(%70, %54) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
              %72 = "arith.maximumf"(%69, %71) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
              "vector.transfer_write"(%72, %24, %arg11, %arg12, %arg13, %arg14) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>}> : (vector<1024xf32>, memref<1x4x4x16xf32>, index, index, index, index) -> ()
              "affine.yield"() : () -> ()
            }) : () -> ()
            "affine.yield"() : () -> ()
          }) : () -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%23) : (memref<1x14x14x16xf32>) -> ()
  %25 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x16x4x4xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg7: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (16)>}> ({
    ^bb0(%arg8: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
      ^bb0(%arg9: index):
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
        ^bb0(%arg10: index):
          %52 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          %53 = "vector.transfer_read"(%24, %arg7, %arg9, %arg10, %arg8, %52) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d0)>}> : (memref<1x4x4x16xf32>, index, index, index, index, f32) -> vector<1024xf32>
          "vector.transfer_write"(%53, %25, %arg7, %arg8, %arg9, %arg10) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 4, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (d0)>}> : (vector<1024xf32>, memref<1x16x4x4xf32>, index, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%24) : (memref<1x4x4x16xf32>) -> ()
  %26 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x1x10xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg3: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
    ^bb0(%arg4: index):
      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (10)>}> ({
      ^bb0(%arg5: index):
        %39 = "arith.constant"() <{value = dense<0.000000e+00> : vector<1024xf32>}> : () -> vector<1024xf32>
        "vector.transfer_write"(%39, %26, %3, %2, %arg5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 3, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d2)>}> : (vector<1024xf32>, memref<1x1x10xf32>, index, index, index) -> ()
        "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (256)>}> ({
        ^bb0(%arg6: index):
          %40 = "affine.apply"(%arg6, %arg3, %arg4) <{map = affine_map<(d0)[s0, s1] -> (s0 + s1)>}> : (index, index, index) -> index
          %41 = "affine.apply"(%arg6, %arg3, %arg4) <{map = affine_map<(d0)[s0, s1] -> (d0 floordiv 16)>}> : (index, index, index) -> index
          %42 = "affine.apply"(%arg6, %arg3, %arg4) <{map = affine_map<(d0)[s0, s1] -> ((d0 mod 16) floordiv 4)>}> : (index, index, index) -> index
          %43 = "affine.apply"(%arg6, %arg3, %arg4) <{map = affine_map<(d0)[s0, s1] -> (d0 mod 4)>}> : (index, index, index) -> index
          %44 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          %45 = "vector.transfer_read"(%25, %40, %41, %42, %43, %44) <{in_bounds = [true], operandSegmentSizes = array<i32: 1, 4, 1, 0>, permutation_map = affine_map<(d0, d1, d2, d3) -> (0)>}> : (memref<1x16x4x4xf32>, index, index, index, index, f32) -> vector<1024xf32>
          %46 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          %47 = "vector.transfer_read"(%13, %arg3, %arg6, %arg5, %46) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d2)>}> : (memref<1x256x10xf32>, index, index, index, f32) -> vector<1024xf32>
          %48 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          %49 = "vector.transfer_read"(%26, %arg3, %arg4, %arg5, %48) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d2)>}> : (memref<1x1x10xf32>, index, index, index, f32) -> vector<1024xf32>
          %50 = "arith.mulf"(%45, %47) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
          %51 = "arith.addf"(%49, %50) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
          "vector.transfer_write"(%51, %26, %arg3, %arg4, %arg5) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 3, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d2)>}> : (vector<1024xf32>, memref<1x1x10xf32>, index, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%25) : (memref<1x16x4x4xf32>) -> ()
  %27 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x10xf32>
  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (1)>}> ({
  ^bb0(%arg1: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1024 : index, upperBoundMap = affine_map<() -> (10)>}> ({
    ^bb0(%arg2: index):
      %29 = "affine.apply"(%arg2) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
      %30 = "affine.apply"(%arg2) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
      %31 = "affine.apply"(%arg2) <{map = affine_map<(d0) -> (d0)>}> : (index) -> index
      %32 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %33 = "vector.transfer_read"(%26, %29, %30, %31, %32) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 3, 1, 0>, permutation_map = affine_map<(d0, d1, d2) -> (d2)>}> : (memref<1x1x10xf32>, index, index, index, f32) -> vector<1024xf32>
      %34 = "affine.apply"(%arg2) <{map = affine_map<(d0) -> (0)>}> : (index) -> index
      %35 = "affine.apply"(%arg2) <{map = affine_map<(d0) -> (d0)>}> : (index) -> index
      %36 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %37 = "vector.transfer_read"(%14, %34, %35, %36) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 2, 1, 0>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (memref<1x10xf32>, index, index, f32) -> vector<1024xf32>
      %38 = "arith.addf"(%33, %37) <{fastmath = #arith.fastmath<none>}> : (vector<1024xf32>, vector<1024xf32>) -> vector<1024xf32>
      "vector.transfer_write"(%38, %27, %arg1, %arg2) <{in_bounds = [false], operandSegmentSizes = array<i32: 1, 1, 2, 0>, permutation_map = affine_map<(d0, d1) -> (d1)>}> : (vector<1024xf32>, memref<1x10xf32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  "memref.dealloc"(%26) : (memref<1x1x10xf32>) -> ()
  %28 = "bufferization.to_tensor"(%27) : (memref<1x10xf32>) -> tensor<1x10xf32>
  "memref.dealloc"(%27) : (memref<1x10xf32>) -> ()
  "func.return"(%28) : (tensor<1x10xf32>) -> ()
}) : () -> ()

