
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
} // transform module