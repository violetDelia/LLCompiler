set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLVMExtension)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLLVMExtensionPassIncGen)
add_dependencies(mlir-headers MLIRLLVMExtensionPassIncGen)

add_mlir_doc(LLVMExtensionPasses LLVMExtensionPasses LLH/ -gen-pass-doc)

