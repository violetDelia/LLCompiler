module @tensor_inlcude attributes { transform.with_named_sequence } {

transform.named_sequence @tensor_basic_opt(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.tensor.decompose_concat
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.tensor.fold_tensor_subset_ops
      transform.apply_patterns.tensor.rewrite_as_constant aggressive
    } : !transform.any_op
    transform.bufferization.buffer_loop_hoisting %funcs : !transform.any_op
    transform.bufferization.eliminate_empty_tensors %funcs : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.yield
  }

}