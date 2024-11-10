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
    // NOTE: unkown assert error -> appending to the MLIRContext dialect registry while in a multi-threaded execution context" while register mhlo-test-unfuse-batch-norm
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