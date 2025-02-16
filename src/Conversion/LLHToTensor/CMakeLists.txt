llcompiler_add_library(LLCLLHToTensor
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
