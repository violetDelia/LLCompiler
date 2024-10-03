llcompiler_add_library(MLIRLLHToTensor
  SRC_FILES
  LLHToTensor.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTensorDialect
  MLIRLLHDialect
  MLIRFuncDialect
)
