# 用transform dialect 构建pass pipeline

## 引言

transform dialect 在mlir在mlir中是十分特殊的dialect，它提供了另一种相比于手写Pass更加细粒度的IR转换的方式。并且可以将转换规则编写成库，像头文件一样引入。这样可以避免经常的修改Pass和Pipeline。虽然用transform dialect 写出的所有变换都可以写成相应的Pass，但是采用transform dialect写要比编写Pass更加直观且高效，而且在一些比较复杂的优化场景上更加灵活。

在mlir的transform dialect实现中具备很多pass【标准】所不具备的一些功能。

## transform dialect 结构

以如下为例：

```mlir
module @tensor_inlcude attributes { transform.with_named_sequence } {

transform.named_sequence @tensor_basic_opt(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op  // 匹配 %module 所有的 func.func op。
    transform.apply_patterns to %funcs { // 对所有的func.func op 依次运行以下变换
      transform.apply_patterns.tensor.decompose_concat
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.tensor.rewrite_as_constant aggressive
    } : !transform.any_op
    transform.yield // 逐一生成并返回
  }
}
```

其中 `transform.with_named_sequence` 表示其实一个描述transform的一个module，它的名字 `tensor_inlcude`。

`transform.named_sequence` 则是在这个module当中的一个优化“规则”，名为 `tensor_basic_opt`，也可以称他为入口名。

优化规则的参数是一个 `!transform.any_op`，其代表任意一种Op，如果要指定Op的类型，可以以这样的形式表示：`!transform.op<"func.func">`。参数后面的属性有两种：

`transform.readonly` 代表这个Op是不能改变的。（可以修改，但是不能重写）

`transform.readonly `则代表Op可以被重写。

在 `transform.named_sequence`  内部则描述着转换的规则。

具体每个Op的含义请移步：[Transform Dialect - MLIR](https://mlir.llvm.org/docs/Dialects/Transform/)

## 运行transform dialect变换

```cpp
mlir::transform::PreloadLibraryPassOptions preload_options;
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_LINALG_INCLUDE__);
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_MHLO_INCLUDE__);
  preload_options.transformLibraryPaths.push_back(
      __LLC_TRANSFORM_TENSOR_INCLUDE__);
  pm.addPass(mlir::transform::createPreloadLibraryPass(preload_options));
```

在mlir中提供了PreloadLibraryPass 用来加载自定义的transfom 文件。如上图所示，输入transform dialect的路径，PassMamager 就会自行管理这些变换规则。

当需要运行某一个运行规则是，调用InterpreterPass 就可以对指定的变换规则进行解释，并在mlir中执行改变换。entry_point 则是运行的入口，也就是定义规则的入口名。所以需要注意的当加载多个transform dialect 文件时，要注意不能有相同的入口名，防止找不到对应的变换规则。

```
void applyInterpreter(::mlir::OpPassManager &pm, const char *entry_point) {
  mlir::transform::InterpreterPassOptions options;
  options.entryPoint = entry_point;
  pm.addPass(mlir::transform::createInterpreterPass(options));
}
```

如果想用mlir-opt工具来运行的话，-transform-preload-library=transform-library-paths=“XXX.mlir” 来加载，用-transform-interpreter=entry-point=“xxx” 来运行。【默认的 entry-point 是 “__transform_main”】

## 示例

下图是一个mhlo的transform dialcet 文件：

```mlir

module @mhlo_inlcude attributes { transform.with_named_sequence } {

transform.named_sequence @mhlo_basic_opt(%module: !transform.any_op {transform.consumed}) {
    transform.apply_patterns to %module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %remove_tuple_funcs = transform.apply_registered_pass "mhlo-flatten-tuple" to %funcs : (!transform.any_op) -> !transform.any_op 
    %conveted_to_signless_module = transform.apply_registered_pass "convert-to-signless" to %module : (!transform.any_op) -> !transform.any_op
    %conveted_to_signless_funcs = transform.structured.match ops{["func.func"]} in %conveted_to_signless_module : (!transform.any_op) -> !transform.any_op
    %simplfy_reduce_funcs = transform.apply_registered_pass "group-reduction-dimensions" to %conveted_to_signless_funcs : (!transform.any_op) -> !transform.any_op
    %simplfy_broadcast_funcs = transform.apply_registered_pass "mhlo-legalize-broadcast-to-broadcast-in-dim" to %simplfy_reduce_funcs : (!transform.any_op) -> !transform.any_op
    %canonicalize_dot_funcs = transform.apply_registered_pass "hlo-canonicalize-dot" to %simplfy_broadcast_funcs : (!transform.any_op) -> !transform.any_op
    %canonicalize_reduce_funcs = transform.apply_registered_pass "hlo-canonicalize-reduction" to %canonicalize_dot_funcs : (!transform.any_op) -> !transform.any_op
    %canonicalize_gather_funcs = transform.apply_registered_pass "hlo-canonicalize-gather" to %canonicalize_reduce_funcs : (!transform.any_op) -> !transform.any_op
    %canonicalize_scatter_funcs = transform.apply_registered_pass "hlo-canonicalize-scatter" to %canonicalize_gather_funcs : (!transform.any_op) -> !transform.any_op
    %cf_sinked_funcs = transform.apply_registered_pass "mhlo-sink-constants-to-control-flow" to %canonicalize_scatter_funcs : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %conveted_to_signless_module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %ops_expanded_funcs = transform.apply_registered_pass "mhlo-expand-ops-simplifier" to %cf_sinked_funcs : (!transform.any_op) -> !transform.any_op
    %batch_norm_decomposed_funcs = transform.apply_registered_pass "mhlo-test-unfuse-batch-norm" to %ops_expanded_funcs : (!transform.any_op) -> !transform.any_op
    %broadcast_sinked_funcs = transform.apply_registered_pass "mhlo-broadcast-propagation" to %batch_norm_decomposed_funcs : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %conveted_to_signless_module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }

transform.named_sequence @mhlo_to_linalg(%module: !transform.any_op {transform.readeonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %opt_shape_funcs = transform.apply_registered_pass "symbolic-shape-optimization" to %funcs : (!transform.any_op) -> !transform.any_op
    %to_std_funcs = transform.apply_registered_pass "mhlo-legalize-to-std" to %opt_shape_funcs : (!transform.any_op) -> !transform.any_op
    %to_linalg_funcs = transform.apply_registered_pass "hlo-legalize-to-linalg" to %to_std_funcs : (!transform.any_op) -> !transform.any_op 
    %lowing_cf = transform.apply_registered_pass "mhlo-legalize-control-flow" to %to_linalg_funcs : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }

transform.named_sequence @mhlo_one_shot_bufferize(%module: !transform.any_op {transform.consumed}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.bufferization.eliminate_empty_tensors %funcs : !transform.any_op
    %empty_ops = transform.structured.match ops{["tensor.empty"]} in %module : (!transform.any_op) -> !transform.op<"tensor.empty">
    transform.bufferization.empty_tensor_to_alloc_tensor %empty_ops : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
    %bufferized_module = transform.bufferization.one_shot_bufferize %module
      {function_boundary_type_conversion = 1 : i32,
      allow_return_allocs_from_loops = true,
      allow_unknown_ops = true,
      bufferize_function_boundaries = true,
      dump_alias_sets = false,
      test_analysis_only = false,
      print_conflicts = false,
      check_parallel_regions = true,
      memcpy_op = "memref.copy"} : (!transform.any_op) -> !transform.any_op
    %bufferized_funcs = transform.structured.match ops{["func.func"]} in %bufferized_module : (!transform.any_op) -> !transform.any_op
    %finnal_funcs = transform.apply_registered_pass "finalizing-bufferize" to %bufferized_funcs  : (!transform.any_op) -> !transform.any_op
    %promote_buffer_module = transform.apply_registered_pass "promote-buffers-to-stack" to %finnal_funcs {options = "max-alloc-size-in-bytes=128"}: (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %promote_buffer_module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
```

mhlo_one_shot_bufferize 这个转换规则表示将抽象的Tensor张量映射到实际的虚拟内存上。

只需要调用该函数就可以运行变换规则。

```cpp
applyInterpreter(pm, "mhlo_one_shot_bufferize");
```

变换前：

```
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  func.func @main(%arg0: tensor<200x3x224x224xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c224"}) -> tensor<200x3x224x224xf32> attributes {entrance} {
    %cst = arith.constant 3.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<200x3x224x224xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<200x3x224x224xf32>) outs(%0 : tensor<200x3x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %in : f32
      %3 = arith.subf %2, %cst : f32
      %4 = arith.divf %3, %cst_0 : f32
      %5 = arith.maximumf %4, %cst_1 : f32
      %6 = arith.mulf %5, %4 : f32
      linalg.yield %6 : f32
    } -> tensor<200x3x224x224xf32>
    return %1 : tensor<200x3x224x224xf32>
  }
  module @__symbol__ {
  }
}
```

变换后：

```
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {builtin.gloabal_layout = #llh.Layout<NCHW>} {
  func.func @main(%arg0: memref<200x3x224x224xf32> {func.input_symbol_0 = "c200", func.input_symbol_1 = "c3", func.input_symbol_2 = "c224", func.input_symbol_3 = "c224"}) -> memref<200x3x224x224xf32> attributes {entrance} {
    %cst = arith.constant 3.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<200x3x224x224xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : memref<200x3x224x224xf32>) outs(%alloc : memref<200x3x224x224xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %in : f32
      %1 = arith.subf %0, %cst : f32
      %2 = arith.divf %1, %cst_0 : f32
      %3 = arith.maximumf %2, %cst_1 : f32
      %4 = arith.mulf %3, %2 : f32
      linalg.yield %4 : f32
    }
    return %alloc : memref<200x3x224x224xf32>
  }
  module @__symbol__ {
  }
}
```
