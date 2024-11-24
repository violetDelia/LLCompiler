set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BufferizationExtension)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LLCBufferizationExtensionPassIncGen)
add_dependencies(mlir-headers LLCBufferizationExtensionPassIncGen)

add_mlir_doc(BufferizationExtensionPasses BufferizationExtensionPasses LLH/ -gen-pass-doc)

