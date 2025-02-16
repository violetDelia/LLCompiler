llcompiler_add_library(LLCLLHToLinalg
  SRC_FILES
  LLHToLinalg.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRArithDialect
  LLCLLHDialect
  MLIRFuncDialect
  MLIRLinalgDialect
)
