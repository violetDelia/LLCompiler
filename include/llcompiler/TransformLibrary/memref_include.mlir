module @memref_include attributes { transform.with_named_sequence } {

  
transform.named_sequence @memref_basic_opt(%module: !transform.any_op {transform.readonly}) {    
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    %add_deallocation_funcs = transform.apply_registered_pass "buffer-deallocation" to %funcs
    : (!transform.any_op) -> !transform.any_op
    %liveness_opted_funcs = transform.apply_registered_pass "optimize-allocation-liveness" to %add_deallocation_funcs : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %liveness_opted_funcs {
        transform.apply_patterns.memref.alloc_to_alloca size_limit(32)
        transform.apply_patterns.memref.resolve_ranked_shaped_type_result_dims
        transform.apply_patterns.memref.expand_strided_metadata
        transform.apply_patterns.canonicalization
    } : !transform.any_op
    %alloca = transform.structured.match ops{["memref.alloca"]} in %module
        : (!transform.any_op) -> !transform.op<"memref.alloca">
    %get_global, %global = transform.memref.alloca_to_global %alloca
          : (!transform.op<"memref.alloca">)
            -> (!transform.any_op, !transform.any_op)
    transform.memref.erase_dead_alloc_and_stores %liveness_opted_funcs : (!transform.any_op) -> ()
    %lowing_affine_funcs = transform.apply_registered_pass "lower-affine" to %liveness_opted_funcs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
} // transform module