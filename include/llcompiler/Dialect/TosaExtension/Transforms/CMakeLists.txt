set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TosaExOpt)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LLCTosaExPassIncGen)
add_dependencies(mlir-headers LLCTosaExPassIncGen)

add_mlir_doc(TosaExPasses TosaExPasses TosaExtension/ -gen-pass-doc)
