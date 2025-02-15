
llcompiler_add_library(LLCLLHTransforms
  SRC_FILES
  TransformLayout.cpp
  LoadWeight.cpp
  Operationlegalization.cpp
  RemoveRedundantOps.cpp
  ReshapeBeforeBraodcast.cpp
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
  LLCLLHDialect
  MLIRPass
  #LLCIRExtension
  LLCMLIRUtility
  LLCLLHUtils
  LLCPatternRewriter
)
