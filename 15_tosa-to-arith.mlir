// -----// IR Dump After TosaToArith (tosa-to-arith) //----- //
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
module attributes {builtin.gloabal_layout = "NCHW"} {
  func.func @main(%arg0: tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>) -> tensor<?x?x?x?xf32> attributes {entrance} {
    %cst = arith.constant dense<0x7F800000> : tensor<1x1x1x1xf32>
    %idx2 = index.constant 2
    %idx1 = index.constant 1
    %idx0 = index.constant 0
    %dim = tensor.dim %arg0, %idx0 : tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>
    %dim_0 = tensor.dim %arg0, %idx1 : tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>
    %dim_1 = tensor.dim %arg0, %idx2 : tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>
    %from_elements = tensor.from_elements %dim, %dim_1, %dim_0, %dim_1 : tensor<4xindex>
    %reshape = tensor.reshape %arg0(%from_elements) : (tensor<?x?x?x?xf32, {"0" = "s0", "1" = "s1", "2" = "s2", "3" = "s2"}>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
    %0 = tensor.empty(%dim_1) : tensor<?xf32>
    %from_elements_2 = tensor.from_elements %idx1, %idx1, %idx1, %dim_1 : tensor<4xindex>
    %reshape_3 = tensor.reshape %0(%from_elements_2) : (tensor<?xf32>, tensor<4xindex>) -> tensor<1x1x1x?xf32>
    %c0 = arith.constant 0 : index
    %dim_4 = tensor.dim %reshape, %c0 : tensor<?x?x?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_5 = tensor.dim %reshape, %c1 : tensor<?x?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_6 = tensor.dim %reshape, %c2 : tensor<?x?x?x?xf32>
    %c3 = arith.constant 3 : index
    %dim_7 = tensor.dim %reshape, %c3 : tensor<?x?x?x?xf32>
    %dim_8 = tensor.dim %reshape_3, %c3 : tensor<1x1x1x?xf32>
    %1 = arith.maxui %dim_7, %dim_8 : index
    %dim_9 = tensor.dim %reshape, %c3 : tensor<?x?x?x?xf32>
    %2 = arith.cmpi eq, %dim_9, %c1 : index
    %3 = scf.if %2 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %reshape, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %reshape, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %reshape, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %1) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %reshape : tensor<?x?x?x?xf32>
    }
    %dim_10 = tensor.dim %reshape_3, %c3 : tensor<1x1x1x?xf32>
    %4 = arith.cmpi eq, %dim_10, %c1 : index
    %5 = scf.if %4 -> (tensor<1x1x1x?xf32>) {
      %86 = tensor.empty(%1) : tensor<1x1x1x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_3 : tensor<1x1x1x?xf32>) outs(%86 : tensor<1x1x1x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x1x?xf32>
      scf.yield %87 : tensor<1x1x1x?xf32>
    } else {
      scf.yield %reshape_3 : tensor<1x1x1x?xf32>
    }
    %6 = tensor.empty(%dim_4, %dim_5, %dim_6, %1) : tensor<?x?x?x?xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %5 : tensor<?x?x?x?xf32>, tensor<1x1x1x?xf32>) outs(%6 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.addf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    %8 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
    %from_elements_11 = tensor.from_elements %idx1, %idx1, %dim_0, %dim_1 : tensor<4xindex>
    %reshape_12 = tensor.reshape %8(%from_elements_11) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<1x1x?x?xf32>
    %c0_13 = arith.constant 0 : index
    %dim_14 = tensor.dim %7, %c0_13 : tensor<?x?x?x?xf32>
    %c1_15 = arith.constant 1 : index
    %dim_16 = tensor.dim %7, %c1_15 : tensor<?x?x?x?xf32>
    %c2_17 = arith.constant 2 : index
    %dim_18 = tensor.dim %7, %c2_17 : tensor<?x?x?x?xf32>
    %dim_19 = tensor.dim %reshape_12, %c2_17 : tensor<1x1x?x?xf32>
    %9 = arith.maxui %dim_18, %dim_19 : index
    %c3_20 = arith.constant 3 : index
    %dim_21 = tensor.dim %7, %c3_20 : tensor<?x?x?x?xf32>
    %dim_22 = tensor.dim %reshape_12, %c3_20 : tensor<1x1x?x?xf32>
    %10 = arith.maxui %dim_21, %dim_22 : index
    %dim_23 = tensor.dim %7, %c2_17 : tensor<?x?x?x?xf32>
    %11 = arith.cmpi eq, %dim_23, %c1_15 : index
    %12 = scf.if %11 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %7, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %7, %c1_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %7, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %9, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %7 : tensor<?x?x?x?xf32>
    }
    %dim_24 = tensor.dim %12, %c3_20 : tensor<?x?x?x?xf32>
    %13 = arith.cmpi eq, %dim_24, %c1_15 : index
    %14 = scf.if %13 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %12, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %12, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %12, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %10) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %12 : tensor<?x?x?x?xf32>
    }
    %dim_25 = tensor.dim %reshape_12, %c2_17 : tensor<1x1x?x?xf32>
    %15 = arith.cmpi eq, %dim_25, %c1_15 : index
    %16 = scf.if %15 -> (tensor<1x1x?x?xf32>) {
      %c3_109 = arith.constant 3 : index
      %dim_110 = tensor.dim %reshape_12, %c3_109 : tensor<1x1x?x?xf32>
      %86 = tensor.empty(%9, %dim_110) : tensor<1x1x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_12 : tensor<1x1x?x?xf32>) outs(%86 : tensor<1x1x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x?x?xf32>
      scf.yield %87 : tensor<1x1x?x?xf32>
    } else {
      scf.yield %reshape_12 : tensor<1x1x?x?xf32>
    }
    %dim_26 = tensor.dim %16, %c3_20 : tensor<1x1x?x?xf32>
    %17 = arith.cmpi eq, %dim_26, %c1_15 : index
    %18 = scf.if %17 -> (tensor<1x1x?x?xf32>) {
      %c2_109 = arith.constant 2 : index
      %dim_110 = tensor.dim %16, %c2_109 : tensor<1x1x?x?xf32>
      %86 = tensor.empty(%dim_110, %10) : tensor<1x1x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<1x1x?x?xf32>) outs(%86 : tensor<1x1x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x?x?xf32>
      scf.yield %87 : tensor<1x1x?x?xf32>
    } else {
      scf.yield %16 : tensor<1x1x?x?xf32>
    }
    %19 = tensor.empty(%dim_14, %dim_16, %9, %10) : tensor<?x?x?x?xf32>
    %20 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14, %18 : tensor<?x?x?x?xf32>, tensor<1x1x?x?xf32>) outs(%19 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.addf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    %21 = tensor.empty(%dim_1) : tensor<?xf32>
    %from_elements_27 = tensor.from_elements %idx1, %idx1, %idx1, %dim_1 : tensor<4xindex>
    %reshape_28 = tensor.reshape %21(%from_elements_27) : (tensor<?xf32>, tensor<4xindex>) -> tensor<1x1x1x?xf32>
    %c0_29 = arith.constant 0 : index
    %dim_30 = tensor.dim %20, %c0_29 : tensor<?x?x?x?xf32>
    %c1_31 = arith.constant 1 : index
    %dim_32 = tensor.dim %20, %c1_31 : tensor<?x?x?x?xf32>
    %c2_33 = arith.constant 2 : index
    %dim_34 = tensor.dim %20, %c2_33 : tensor<?x?x?x?xf32>
    %c3_35 = arith.constant 3 : index
    %dim_36 = tensor.dim %20, %c3_35 : tensor<?x?x?x?xf32>
    %dim_37 = tensor.dim %reshape_28, %c3_35 : tensor<1x1x1x?xf32>
    %22 = arith.maxui %dim_36, %dim_37 : index
    %dim_38 = tensor.dim %20, %c3_35 : tensor<?x?x?x?xf32>
    %23 = arith.cmpi eq, %dim_38, %c1_31 : index
    %24 = scf.if %23 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %20, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %20, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %20, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %22) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %20 : tensor<?x?x?x?xf32>
    }
    %dim_39 = tensor.dim %reshape_28, %c3_35 : tensor<1x1x1x?xf32>
    %25 = arith.cmpi eq, %dim_39, %c1_31 : index
    %26 = scf.if %25 -> (tensor<1x1x1x?xf32>) {
      %86 = tensor.empty(%22) : tensor<1x1x1x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_28 : tensor<1x1x1x?xf32>) outs(%86 : tensor<1x1x1x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x1x?xf32>
      scf.yield %87 : tensor<1x1x1x?xf32>
    } else {
      scf.yield %reshape_28 : tensor<1x1x1x?xf32>
    }
    %27 = tensor.empty(%dim_30, %dim_32, %dim_34, %22) : tensor<?x?x?x?xf32>
    %28 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %26 : tensor<?x?x?x?xf32>, tensor<1x1x1x?xf32>) outs(%27 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.subf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    %29 = tensor.empty(%dim_0, %dim_1) : tensor<?x?xf32>
    %from_elements_40 = tensor.from_elements %idx1, %idx1, %dim_0, %dim_1 : tensor<4xindex>
    %reshape_41 = tensor.reshape %29(%from_elements_40) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<1x1x?x?xf32>
    %c0_42 = arith.constant 0 : index
    %dim_43 = tensor.dim %28, %c0_42 : tensor<?x?x?x?xf32>
    %c1_44 = arith.constant 1 : index
    %dim_45 = tensor.dim %28, %c1_44 : tensor<?x?x?x?xf32>
    %c2_46 = arith.constant 2 : index
    %dim_47 = tensor.dim %28, %c2_46 : tensor<?x?x?x?xf32>
    %dim_48 = tensor.dim %reshape_41, %c2_46 : tensor<1x1x?x?xf32>
    %30 = arith.maxui %dim_47, %dim_48 : index
    %c3_49 = arith.constant 3 : index
    %dim_50 = tensor.dim %28, %c3_49 : tensor<?x?x?x?xf32>
    %dim_51 = tensor.dim %reshape_41, %c3_49 : tensor<1x1x?x?xf32>
    %31 = arith.maxui %dim_50, %dim_51 : index
    %dim_52 = tensor.dim %28, %c2_46 : tensor<?x?x?x?xf32>
    %32 = arith.cmpi eq, %dim_52, %c1_44 : index
    %33 = scf.if %32 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %28, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %28, %c1_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %28, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %30, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %28 : tensor<?x?x?x?xf32>
    }
    %dim_53 = tensor.dim %33, %c3_49 : tensor<?x?x?x?xf32>
    %34 = arith.cmpi eq, %dim_53, %c1_44 : index
    %35 = scf.if %34 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %33, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %33, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %33, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %31) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %33 : tensor<?x?x?x?xf32>
    }
    %dim_54 = tensor.dim %reshape_41, %c2_46 : tensor<1x1x?x?xf32>
    %36 = arith.cmpi eq, %dim_54, %c1_44 : index
    %37 = scf.if %36 -> (tensor<1x1x?x?xf32>) {
      %c3_109 = arith.constant 3 : index
      %dim_110 = tensor.dim %reshape_41, %c3_109 : tensor<1x1x?x?xf32>
      %86 = tensor.empty(%30, %dim_110) : tensor<1x1x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_41 : tensor<1x1x?x?xf32>) outs(%86 : tensor<1x1x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x?x?xf32>
      scf.yield %87 : tensor<1x1x?x?xf32>
    } else {
      scf.yield %reshape_41 : tensor<1x1x?x?xf32>
    }
    %dim_55 = tensor.dim %37, %c3_49 : tensor<1x1x?x?xf32>
    %38 = arith.cmpi eq, %dim_55, %c1_44 : index
    %39 = scf.if %38 -> (tensor<1x1x?x?xf32>) {
      %c2_109 = arith.constant 2 : index
      %dim_110 = tensor.dim %37, %c2_109 : tensor<1x1x?x?xf32>
      %86 = tensor.empty(%dim_110, %31) : tensor<1x1x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%37 : tensor<1x1x?x?xf32>) outs(%86 : tensor<1x1x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x1x?x?xf32>
      scf.yield %87 : tensor<1x1x?x?xf32>
    } else {
      scf.yield %37 : tensor<1x1x?x?xf32>
    }
    %40 = tensor.empty(%dim_43, %dim_45, %30, %31) : tensor<?x?x?x?xf32>
    %41 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %39 : tensor<?x?x?x?xf32>, tensor<1x1x?x?xf32>) outs(%40 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.mulf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    %42 = tensor.empty(%dim_1, %dim_0, %dim_1) : tensor<?x?x?xf32>
    %from_elements_56 = tensor.from_elements %idx1, %dim_1, %dim_0, %dim_1 : tensor<4xindex>
    %reshape_57 = tensor.reshape %42(%from_elements_56) : (tensor<?x?x?xf32>, tensor<4xindex>) -> tensor<1x?x?x?xf32>
    %c1_58 = arith.constant 1 : index
    %dim_59 = tensor.dim %reshape_57, %c1_58 : tensor<1x?x?x?xf32>
    %c2_60 = arith.constant 2 : index
    %dim_61 = tensor.dim %reshape_57, %c2_60 : tensor<1x?x?x?xf32>
    %c3_62 = arith.constant 3 : index
    %dim_63 = tensor.dim %reshape_57, %c3_62 : tensor<1x?x?x?xf32>
    %43 = tensor.empty(%dim_59, %dim_61, %dim_63) : tensor<1x?x?x?xf32>
    %44 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_57 : tensor<1x?x?x?xf32>) outs(%43 : tensor<1x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_109 = arith.constant 1.000000e+00 : f32
      %86 = arith.divf %cst_109, %in : f32
      linalg.yield %86 : f32
    } -> tensor<1x?x?x?xf32>
    %c0_64 = arith.constant 0 : index
    %dim_65 = tensor.dim %41, %c0_64 : tensor<?x?x?x?xf32>
    %c1_66 = arith.constant 1 : index
    %dim_67 = tensor.dim %41, %c1_66 : tensor<?x?x?x?xf32>
    %dim_68 = tensor.dim %44, %c1_66 : tensor<1x?x?x?xf32>
    %45 = arith.maxui %dim_67, %dim_68 : index
    %c2_69 = arith.constant 2 : index
    %dim_70 = tensor.dim %41, %c2_69 : tensor<?x?x?x?xf32>
    %dim_71 = tensor.dim %44, %c2_69 : tensor<1x?x?x?xf32>
    %46 = arith.maxui %dim_70, %dim_71 : index
    %c3_72 = arith.constant 3 : index
    %dim_73 = tensor.dim %41, %c3_72 : tensor<?x?x?x?xf32>
    %dim_74 = tensor.dim %44, %c3_72 : tensor<1x?x?x?xf32>
    %47 = arith.maxui %dim_73, %dim_74 : index
    %dim_75 = tensor.dim %41, %c1_66 : tensor<?x?x?x?xf32>
    %48 = arith.cmpi eq, %dim_75, %c1_66 : index
    %49 = scf.if %48 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %41, %c0_109 : tensor<?x?x?x?xf32>
      %c2_111 = arith.constant 2 : index
      %dim_112 = tensor.dim %41, %c2_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %41, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %45, %dim_112, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %41 : tensor<?x?x?x?xf32>
    }
    %dim_76 = tensor.dim %49, %c2_69 : tensor<?x?x?x?xf32>
    %50 = arith.cmpi eq, %dim_76, %c1_66 : index
    %51 = scf.if %50 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %49, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %49, %c1_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %49, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %46, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%49 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %49 : tensor<?x?x?x?xf32>
    }
    %dim_77 = tensor.dim %51, %c3_72 : tensor<?x?x?x?xf32>
    %52 = arith.cmpi eq, %dim_77, %c1_66 : index
    %53 = scf.if %52 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %51, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %51, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %51, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %47) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %51 : tensor<?x?x?x?xf32>
    }
    %dim_78 = tensor.dim %44, %c1_66 : tensor<1x?x?x?xf32>
    %54 = arith.cmpi eq, %dim_78, %c1_66 : index
    %55 = scf.if %54 -> (tensor<1x?x?x?xf32>) {
      %c2_109 = arith.constant 2 : index
      %dim_110 = tensor.dim %44, %c2_109 : tensor<1x?x?x?xf32>
      %c3_111 = arith.constant 3 : index
      %dim_112 = tensor.dim %44, %c3_111 : tensor<1x?x?x?xf32>
      %86 = tensor.empty(%45, %dim_110, %dim_112) : tensor<1x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%44 : tensor<1x?x?x?xf32>) outs(%86 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x?x?x?xf32>
      scf.yield %87 : tensor<1x?x?x?xf32>
    } else {
      scf.yield %44 : tensor<1x?x?x?xf32>
    }
    %dim_79 = tensor.dim %55, %c2_69 : tensor<1x?x?x?xf32>
    %56 = arith.cmpi eq, %dim_79, %c1_66 : index
    %57 = scf.if %56 -> (tensor<1x?x?x?xf32>) {
      %c1_109 = arith.constant 1 : index
      %dim_110 = tensor.dim %55, %c1_109 : tensor<1x?x?x?xf32>
      %c3_111 = arith.constant 3 : index
      %dim_112 = tensor.dim %55, %c3_111 : tensor<1x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %46, %dim_112) : tensor<1x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%55 : tensor<1x?x?x?xf32>) outs(%86 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x?x?x?xf32>
      scf.yield %87 : tensor<1x?x?x?xf32>
    } else {
      scf.yield %55 : tensor<1x?x?x?xf32>
    }
    %dim_80 = tensor.dim %57, %c3_72 : tensor<1x?x?x?xf32>
    %58 = arith.cmpi eq, %dim_80, %c1_66 : index
    %59 = scf.if %58 -> (tensor<1x?x?x?xf32>) {
      %c1_109 = arith.constant 1 : index
      %dim_110 = tensor.dim %57, %c1_109 : tensor<1x?x?x?xf32>
      %c2_111 = arith.constant 2 : index
      %dim_112 = tensor.dim %57, %c2_111 : tensor<1x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %47) : tensor<1x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57 : tensor<1x?x?x?xf32>) outs(%86 : tensor<1x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x?x?x?xf32>
      scf.yield %87 : tensor<1x?x?x?xf32>
    } else {
      scf.yield %57 : tensor<1x?x?x?xf32>
    }
    %60 = tensor.empty(%dim_65, %45, %46, %47) : tensor<?x?x?x?xf32>
    %61 = linalg.generic {indexing_maps = [#map1, #map5, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %59 : tensor<?x?x?x?xf32>, tensor<1x?x?x?xf32>) outs(%60 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.mulf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    %c0_81 = arith.constant 0 : index
    %dim_82 = tensor.dim %61, %c0_81 : tensor<?x?x?x?xf32>
    %dim_83 = tensor.dim %61, %c0_81 : tensor<?x?x?x?xf32>
    %62 = arith.maxui %dim_82, %dim_83 : index
    %c1_84 = arith.constant 1 : index
    %dim_85 = tensor.dim %61, %c1_84 : tensor<?x?x?x?xf32>
    %dim_86 = tensor.dim %61, %c1_84 : tensor<?x?x?x?xf32>
    %63 = arith.maxui %dim_85, %dim_86 : index
    %c2_87 = arith.constant 2 : index
    %dim_88 = tensor.dim %61, %c2_87 : tensor<?x?x?x?xf32>
    %dim_89 = tensor.dim %61, %c2_87 : tensor<?x?x?x?xf32>
    %64 = arith.maxui %dim_88, %dim_89 : index
    %c3_90 = arith.constant 3 : index
    %dim_91 = tensor.dim %61, %c3_90 : tensor<?x?x?x?xf32>
    %dim_92 = tensor.dim %61, %c3_90 : tensor<?x?x?x?xf32>
    %65 = arith.maxui %dim_91, %dim_92 : index
    %dim_93 = tensor.dim %61, %c0_81 : tensor<?x?x?x?xf32>
    %66 = arith.cmpi eq, %dim_93, %c1_84 : index
    %67 = scf.if %66 -> (tensor<?x?x?x?xf32>) {
      %c1_109 = arith.constant 1 : index
      %dim_110 = tensor.dim %61, %c1_109 : tensor<?x?x?x?xf32>
      %c2_111 = arith.constant 2 : index
      %dim_112 = tensor.dim %61, %c2_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %61, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%62, %dim_110, %dim_112, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %61 : tensor<?x?x?x?xf32>
    }
    %dim_94 = tensor.dim %67, %c1_84 : tensor<?x?x?x?xf32>
    %68 = arith.cmpi eq, %dim_94, %c1_84 : index
    %69 = scf.if %68 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %67, %c0_109 : tensor<?x?x?x?xf32>
      %c2_111 = arith.constant 2 : index
      %dim_112 = tensor.dim %67, %c2_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %67, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %63, %dim_112, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %67 : tensor<?x?x?x?xf32>
    }
    %dim_95 = tensor.dim %69, %c2_87 : tensor<?x?x?x?xf32>
    %70 = arith.cmpi eq, %dim_95, %c1_84 : index
    %71 = scf.if %70 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %69, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %69, %c1_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %69, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %64, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %69 : tensor<?x?x?x?xf32>
    }
    %dim_96 = tensor.dim %71, %c3_90 : tensor<?x?x?x?xf32>
    %72 = arith.cmpi eq, %dim_96, %c1_84 : index
    %73 = scf.if %72 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %71, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %71, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %71, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %65) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %71 : tensor<?x?x?x?xf32>
    }
    %dim_97 = tensor.dim %61, %c0_81 : tensor<?x?x?x?xf32>
    %74 = arith.cmpi eq, %dim_97, %c1_84 : index
    %75 = scf.if %74 -> (tensor<?x?x?x?xf32>) {
      %c1_109 = arith.constant 1 : index
      %dim_110 = tensor.dim %61, %c1_109 : tensor<?x?x?x?xf32>
      %c2_111 = arith.constant 2 : index
      %dim_112 = tensor.dim %61, %c2_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %61, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%62, %dim_110, %dim_112, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map5, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %61 : tensor<?x?x?x?xf32>
    }
    %dim_98 = tensor.dim %75, %c1_84 : tensor<?x?x?x?xf32>
    %76 = arith.cmpi eq, %dim_98, %c1_84 : index
    %77 = scf.if %76 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %75, %c0_109 : tensor<?x?x?x?xf32>
      %c2_111 = arith.constant 2 : index
      %dim_112 = tensor.dim %75, %c2_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %75, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %63, %dim_112, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %75 : tensor<?x?x?x?xf32>
    }
    %dim_99 = tensor.dim %77, %c2_87 : tensor<?x?x?x?xf32>
    %78 = arith.cmpi eq, %dim_99, %c1_84 : index
    %79 = scf.if %78 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %77, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %77, %c1_111 : tensor<?x?x?x?xf32>
      %c3_113 = arith.constant 3 : index
      %dim_114 = tensor.dim %77, %c3_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %64, %dim_114) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%77 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %77 : tensor<?x?x?x?xf32>
    }
    %dim_100 = tensor.dim %79, %c3_90 : tensor<?x?x?x?xf32>
    %80 = arith.cmpi eq, %dim_100, %c1_84 : index
    %81 = scf.if %80 -> (tensor<?x?x?x?xf32>) {
      %c0_109 = arith.constant 0 : index
      %dim_110 = tensor.dim %79, %c0_109 : tensor<?x?x?x?xf32>
      %c1_111 = arith.constant 1 : index
      %dim_112 = tensor.dim %79, %c1_111 : tensor<?x?x?x?xf32>
      %c2_113 = arith.constant 2 : index
      %dim_114 = tensor.dim %79, %c2_113 : tensor<?x?x?x?xf32>
      %86 = tensor.empty(%dim_110, %dim_112, %dim_114, %65) : tensor<?x?x?x?xf32>
      %87 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79 : tensor<?x?x?x?xf32>) outs(%86 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?x?x?xf32>
      scf.yield %87 : tensor<?x?x?x?xf32>
    } else {
      scf.yield %79 : tensor<?x?x?x?xf32>
    }
    %82 = tensor.empty(%62, %63, %64, %65) : tensor<?x?x?x?xf32>
    %83 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %81 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%82 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.mulf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    %c0_101 = arith.constant 0 : index
    %dim_102 = tensor.dim %83, %c0_101 : tensor<?x?x?x?xf32>
    %c1_103 = arith.constant 1 : index
    %dim_104 = tensor.dim %83, %c1_103 : tensor<?x?x?x?xf32>
    %c2_105 = arith.constant 2 : index
    %dim_106 = tensor.dim %83, %c2_105 : tensor<?x?x?x?xf32>
    %c3_107 = arith.constant 3 : index
    %dim_108 = tensor.dim %83, %c3_107 : tensor<?x?x?x?xf32>
    %84 = tensor.empty(%dim_102, %dim_104, %dim_106, %dim_108) : tensor<?x?x?x?xf32>
    %85 = linalg.generic {indexing_maps = [#map1, #map7, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%83, %cst : tensor<?x?x?x?xf32>, tensor<1x1x1x1xf32>) outs(%84 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_109: f32, %out: f32):
      %86 = arith.mulf %in, %in_109 : f32
      linalg.yield %86 : f32
    } -> tensor<?x?x?x?xf32>
    return %85 : tensor<?x?x?x?xf32>
  }
}


