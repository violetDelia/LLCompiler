llcompiler_add_library(LLCLLHToFunc
  SRC_FILES
  LLHToFunc.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRArithDialect
  LLCLLHDialect
  MLIRFuncDialect
  MLIRMemRefDialect
)