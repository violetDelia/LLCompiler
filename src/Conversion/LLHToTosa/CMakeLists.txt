llcompiler_add_library(LLCLLHToTosa
  SRC_FILES
  LLHToTosa.cpp

  DEPENDS
  LLCConversionPassIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTosaDialect
  LLCLLHDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
)


