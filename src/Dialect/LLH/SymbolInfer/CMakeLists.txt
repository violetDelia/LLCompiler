
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
  LLCSymbolInferPassIncGen
  LLHTransformLayoutToNHWCIncGen

  LINKS PUBLIC
  LLCLLHDialect
  MLIRPass
  LLCLLHDialect
  MLIRArithDialect
  MLIRTensorDialect
)
