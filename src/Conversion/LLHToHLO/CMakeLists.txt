set(LLVM_TARGET_DEFINITIONS LLHToHLO.td)
mlir_tablegen(LLHToHLO.inc -gen-rewriters)
add_public_tablegen_target(LLCLLHToHLOIncGen)

llcompiler_add_library(LLCLLHToHLO
  SRC_FILES
  LLHToHLO.cpp
  LLHPreprocessingForHLO.cpp

  DEPENDS
  LLCConversionPassIncGen
  LLCLLHToHLOIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTensorDialect
  MLIRLLHDialect
  MLIRFuncDialect
  StablehloOps
)
