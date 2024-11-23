llcompiler_add_library(LLCLLHToShape
  SRC_FILES
  LLHToShape.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRLLHDialect
  MLIRFuncDialect
  MLIRShapeDialect
)
