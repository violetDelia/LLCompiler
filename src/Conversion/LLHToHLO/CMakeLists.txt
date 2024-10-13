llcompiler_add_library(MLIRLLHToHLO
  SRC_FILES
  LLHToHLO.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTensorDialect
  MLIRLLHDialect
  MLIRFuncDialect
  StablehloOps
)
