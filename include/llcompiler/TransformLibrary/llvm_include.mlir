
module @llvm_include attributes { transform.with_named_sequence } {

  
transform.named_sequence @lowing_to_llvm(%module: !transform.any_op {transform.consumed}) {
  %lowing_vector_module = transform.apply_registered_pass "convert-vector-to-llvm" to %module {options = "reassociate-fp-reductions force-32bit-vector-indices=0"}: (!transform.any_op) -> !transform.any_op 
  %lowing_func_module = transform.apply_registered_pass "convert-func-to-llvm" to %lowing_vector_module {options = "index-bitwidth=64"}: (!transform.any_op) -> !transform.any_op 
  %funcs = transform.structured.match ops{["llvm.func"]} in %lowing_func_module
    : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %funcs {
    transform.apply_conversion_patterns.dialect_to_llvm "math"
    transform.apply_conversion_patterns.dialect_to_llvm "memref"
    transform.apply_conversion_patterns.dialect_to_llvm "index"
    transform.apply_conversion_patterns.dialect_to_llvm "arith"
    transform.apply_conversion_patterns.dialect_to_llvm "cf"
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
      {index_bitwidth = 64,
       use_bare_ptr = false,
       use_bare_ptr_memref_call_conv = false,
       use_opaque_pointers = true}
  } {
    legal_dialects = ["llvm"],
    partial_conversion
  } : !transform.any_op
  %funcs_1 = transform.structured.match ops{["llvm.func"]} in %lowing_func_module
    : (!transform.any_op) -> !transform.any_op
  %final = transform.apply_registered_pass "reconcile-unrealized-casts" to %funcs_1
    : (!transform.any_op) -> !transform.any_op
  transform.yield
  }

transform.named_sequence @llvm_basic_opt(%module: !transform.any_op {transform.consumed}) {
    %adpated_args_module = transform.apply_registered_pass "adapt-entry-parms-for-engine" to %module : (!transform.any_op) -> !transform.any_op 
    %legalized_module = transform.apply_registered_pass "llvm-legalize-for-export" to %adpated_args_module : (!transform.any_op) -> !transform.any_op 
    %alloca_opted_module = transform.apply_registered_pass "mem2reg" to %legalized_module : (!transform.any_op) -> !transform.any_op 
    transform.apply_dce to %alloca_opted_module : !transform.any_op
    transform.apply_cse to %alloca_opted_module : !transform.any_op
    transform.apply_patterns to %alloca_opted_module {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }  
} // transform module