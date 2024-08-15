add_mlir_dialect_library(LLCCommonPipeline
    CommonPipeline.cpp
    Enums.cpp

    ADDITIONAL_HEADER_DIRS
    ${LLCOMPILER_INCLUDE_DIR}/llcompiler/Pipeline

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransforms
    MLIRLLHToTosa
    MLIRTosaTransforms
    MLIRTosaToArith
    MLIRTosaToTensor
    MLIRTosaToLinalg
    MLIRTosaToSCF
)

llcompiler_install_mlir_library(LLCCommonPipeline PIPELINE)