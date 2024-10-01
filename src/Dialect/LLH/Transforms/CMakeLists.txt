llcompiler_add_library(MLIRLLHTransforms
  SRC_FILES
  TransformLayoutToNHWC.cpp
  LoadWeight.cpp
  InferSymbolShape.cpp
  Operationlegalization.cpp
  RemoveRedundantOps.cpp
  ReshapeBeforeBraodcast.cpp

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
  MLIRSymbolShapeOpInterfaces
  LLCPatternRewriter
)

# add_mlir_library(MLIRLLHTransforms
# TransformLayoutToNHWC.cpp

# ADDITIONAL_HEADER_DIRS
# ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH/Transforms

# DEPENDS
# MLIRLLHDialectIncGen
# MLIRLLHPassIncGen
# LLHTransformLayoutToNHWCIncGen

# LINK_LIBS PUBLIC
# MLIRLLHDialect
# MLIRPass
# MLIRIRExtension
# LLCMLIRUtility
# )
# llcompiler_install_mlir_target(MLIRLLHTransforms TRANSFORM)