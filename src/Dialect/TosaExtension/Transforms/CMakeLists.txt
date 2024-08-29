add_mlir_library(MLIRTosaExTransforms
  TransformLayoutToNHWC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/TosaExtension/Transforms

  DEPENDS
  MLIRTosaExDialectIncGen
  MLIRTosaExPassIncGen

  LINK_LIBS PUBLIC
  MLIRTosaExDialect
  MLIRPass
)
llcompiler_install_mlir_target(MLIRTosaExTransforms TRANSFORM)