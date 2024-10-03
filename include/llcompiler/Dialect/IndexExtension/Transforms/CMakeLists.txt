set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name IndexExtension)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRIndexExtensionPassIncGen)
add_dependencies(mlir-headers MLIRIndexExtensionPassIncGen)

add_mlir_doc(IndexExtensionPasses IndexExtensionPasses LLH/ -gen-pass-doc)

