llcompiler_add_library(MLIRLLHToTosa
  SRC_FILES
  LLHToTosa.cpp

  DEPENDS
  LLCConversionPassIncGen
  MLIRLLHToTosaIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTosaDialect
  MLIRLLHDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
)


