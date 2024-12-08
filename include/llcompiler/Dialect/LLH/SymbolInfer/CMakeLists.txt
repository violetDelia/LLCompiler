set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLCSymbolOpt)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LLCSymbolInferPassIncGen)
add_dependencies(mlir-headers LLCSymbolInferPassIncGen)

add_mlir_doc(SymbolInfer SymbolInfer LLH/ -gen-pass-doc)

