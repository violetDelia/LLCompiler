
llcompiler_add_library(LLCLLHTransforms
  SRC_FILES
  TransformLayout.cpp
  LoadWeight.cpp
  InferSymbolShape.cpp
  Operationlegalization.cpp
  RemoveRedundantOps.cpp
  ReshapeBeforeBraodcast.cpp
  UnloadAndBindEncoding.cpp
  SinkBindEncoding.cpp
  RemoveSymbol.cpp
  InsetBraodCastPass.cpp
  MarkAot.cpp
  DecomposeOps.cpp

  ADDITIONAL_INCLUDE_DIRS
  ${SYMENGINE_INCLUDE_DIRS}

  DEPENDS
  LLCLLHDialectIncGen
  LLCLLHPassIncGen
  LLHTransformLayoutToNHWCIncGen

  LINKS PUBLIC
  MLIRLLHDialect
  MLIRPass
  LLCIRExtension
  LLCMLIRUtility
  LLCLLHUtils
  LLCPatternRewriter
)
