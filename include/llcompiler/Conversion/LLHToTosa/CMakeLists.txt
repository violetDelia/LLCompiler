set(LLVM_TARGET_DEFINITIONS LLHToTosa.td)
mlir_tablegen(LLHToTosa.inc -gen-rewriters)
add_public_tablegen_target(MLIRLLHToTosaIncGen)