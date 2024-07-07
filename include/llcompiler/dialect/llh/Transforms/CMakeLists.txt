set(LLVM_TARGET_DEFINITIONS LLHPasses.td)
mlir_tablegen(LLHPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRLLHPassesIncGen)

add_mlir_doc(LLHPasses LLHPasses LLH/ -gen-pass-doc)