llcompiler_add_library(MLIRLLHToTosa
  SRC_FILES
  LLHToTosa.cpp

  DEPENDS
  LLCConversionPassIncGen
  MLIRLLHToTosaIncGen

  LINKS PUBLIC
  LLCMLIRUtility
  MLIRIR
  MLIRTosaDialect
  MLIRLLHDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
)

# add_mlir_conversion_library(MLIRLLHToTosa
# LLHToTosa.cpp

# ADDITIONAL_HEADER_DIRS
# ${LLCOMPILER_INCLUDE_DIR}/llcompiler/Conversion/LLHToTosa

# DEPENDS
# LLCConversionPassIncGen
# MLIRLLHToTosaIncGen

# LINK_LIBS PUBLIC
# LLCMLIRUtility
# MLIRIR
# MLIRTosaDialect
# MLIRLLHDialect
# MLIRFuncDialect
# MLIRPass
# MLIRTransformUtils
# )

# llcompiler_install_mlir_target(MLIRLLHToTosa CONVERSION)
