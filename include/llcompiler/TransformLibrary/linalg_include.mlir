
module @linalg_include attributes { transform.with_named_sequence } {

  
transform.named_sequence @linalg_generalize(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module : (!transform.any_op) -> !transform.any_op
    %generalize_ops = transform.structured.generalize %linalg_ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @linalg_specialize(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module : (!transform.any_op) -> !transform.any_op
    %specialize_ops = transform.structured.specialize %linalg_ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @linalg_flatten_elementwise(%module: !transform.any_op {transform.readonly}) {
    %linalg_ops = transform.structured.match interface{LinalgOp} in %module : (!transform.any_op) -> !transform.any_op
    %specialize_ops = transform.structured.flatten_elementwise %linalg_ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

transform.named_sequence @vector_linalg(%func: !transform.op<"func.func"> {transform.consumed}) {
    %generic_ops = transform.structured.match ops{["linalg.generic"]} in %func
      : (!transform.op<"func.func">) -> !transform.any_op
    transform.structured.vectorize %generic_ops vector_sizes [[4]] : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.op<"func.func">

    %func_h = transform.structured.hoist_redundant_vector_transfers %func
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    %all_loops = transform.structured.match interface{LoopLikeInterface} in %func_h
      : (!transform.op<"func.func">) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

    transform.apply_patterns to %func_h {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    transform.apply_patterns to %func_h {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.op<"func.func">
    transform.yield
  }

} // transform module