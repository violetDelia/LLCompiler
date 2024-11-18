
module @scf_include attributes { transform.with_named_sequence } {

  
transform.named_sequence @loop_pipeline(%module: !transform.any_op {transform.readonly}) {
    %scf_for_ops = transform.structured.match ops{["scf.for"]} in %module : (!transform.any_op) -> !transform.op<"scf.for">
    %1 = transform.loop.pipeline %scf_for_ops {iteration_interval = 1 : i64, read_latency = 5 : i64,  scheduling_type = "full-loops"} : (!transform.op<"scf.for">) -> !transform.any_op
     transform.yield
 }
}