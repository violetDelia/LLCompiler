add_mlir_library(MLIRLLHTransforms
  TransformLayoutToNHWC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/LLH/Transforms

  DEPENDS
  MLIRLLHDialectIncGen
  MLIRLLHPassIncGen
  LLHTransformLayoutToNHWCIncGen

  LINK_LIBS PUBLIC
  MLIRLLHDialect
  MLIRPass
  MLIRIRExtension
  LLCMLIRUtility
)
llcompiler_install_mlir_library(MLIRLLHTransforms TRANSFORM)