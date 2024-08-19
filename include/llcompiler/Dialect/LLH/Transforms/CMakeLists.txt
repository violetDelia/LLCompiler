set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLHOpt)
mlir_tablegen(PassesEnums.h.inc -gen-enum-decls)
mlir_tablegen(PassesEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLLHPassIncGen)
add_dependencies(mlir-headers MLIRLLHPassIncGen)

add_mlir_doc(LLHPasses LLHPasses LLH/ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS TransformLayoutToNHWC.td)
mlir_tablegen(TransformLayoutToNHWC.inc -gen-rewriters)
add_public_tablegen_target(LLHTransformLayoutToNHWCIncGen)
