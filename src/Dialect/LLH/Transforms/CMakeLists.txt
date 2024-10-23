
llcompiler_add_library(MLIRLLHTransforms
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
  MLIRIRExtension
  LLCMLIRUtility
  MLIRLLHUtils
  LLCPatternRewriter
)
