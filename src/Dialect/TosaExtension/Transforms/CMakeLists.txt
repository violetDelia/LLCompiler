llcompiler_add_library(LLCTosaExtensionTransforms
  SRC_FILES
  TransformLayoutToNHWC.cpp

  DEPENDS
  LLCTosaExDialectIncGen
  LLCTosaExPassIncGen

  LINKS PUBLIC
  MLIRTosaExDialect
  MLIRPass
)

# add_mlir_library(LLCTosaExtensionTransforms
# TransformLayoutToNHWC.cpp

# ADDITIONAL_HEADER_DIRS
# ${PROJECT_SOURCE_DIR}/include/llcompiler/Dialect/TosaExtension/Transforms

# DEPENDS
# LLCTosaExDialectIncGen
# LLCTosaExPassIncGen

# LINK_LIBS PUBLIC
# MLIRTosaExDialect
# MLIRPass
# )
# llcompiler_install_mlir_target(LLCTosaExtensionTransforms TRANSFORM)