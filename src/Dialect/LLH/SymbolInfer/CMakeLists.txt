
llcompiler_add_library(LLCSymbolInferTransforms
  SRC_FILES
  InferSymbolShape.cpp
  RemoveSymbol.cpp
  SinkBindEncoding.cpp
  UnloadAndBindEncoding.cpp
  SymbolCSE.cpp



  ADDITIONAL_INCLUDE_DIRS
  ${SYMENGINE_INCLUDE_DIRS}

  DEPENDS
  LLCLLHDialectIncGen
  LLCLLHPassIncGen
  LLHTransformLayoutToNHWCIncGen

  LINKS PUBLIC
  MLIRLLHDialect
  MLIRPass
  MLIRLLHDialect
  MLIRArithDialect
  MLIRTensorDialect
)
