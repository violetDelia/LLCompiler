
module @llh_optimization attributes { transform.with_named_sequence } {

  
transform.named_sequence @llh_optimization(%module: !transform.any_op {transform.consumed}) -> !transform.any_op {
  %aot_marked = transform.apply_registered_pass "mark-aot" to %module : (!transform.any_op) -> !transform.any_op
  %redundant_removed = transform.apply_registered_pass "remove-redundant-ops" to %aot_marked : (!transform.any_op) -> !transform.any_op
  %inlined = transform.apply_registered_pass "inline" to %redundant_removed : (!transform.any_op) -> !transform.any_op
  %inlined = transform.apply_registered_pass "infer-symbol-shape" to %inlined : (!transform.any_op) -> !transform.any_op
  transform.yield %module : !transform.any_op
}
} // transform module