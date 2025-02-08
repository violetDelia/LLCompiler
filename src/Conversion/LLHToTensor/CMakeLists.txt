llcompiler_add_library(LLCCLLHToTensor
  SRC_FILES
  LLHToTensor.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTensorDialect
  LLCLLHDialect
  MLIRFuncDialect
)
