llcompiler_add_library(MLIRTosaToLinalgExtension
  SRC_FILES
  TosaToLinalgExtension.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRLinalgDialect
  LLCLLHDialect
  MLIRFuncDialect
)
