add_subdirectory(LLHToTosa)
add_subdirectory(LLHToTensor)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLCConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix LLCConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix LLCConversion)
add_public_tablegen_target(LLCConversionPassIncGen)
add_mlir_doc(LLCPasses LLCConversionPasses LLC/Conversion/ -gen-pass-doc)