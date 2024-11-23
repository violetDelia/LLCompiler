
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

  DEPENDS
  MLIRLLHDialectIncGen
  MLIRLLHPassIncGen
  LLHTransformLayoutToNHWCIncGen
  

  LINKS PUBLIC
  MLIRLLHDialect
  MLIRPass
  LLCIRExtension
  LLCMLIRUtility
  LLCLLHUtils
  LLCPatternRewriter
)
