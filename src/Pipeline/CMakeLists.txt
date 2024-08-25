add_mlir_dialect_library(LLCCommonPipeline
    CommonPipeline.cpp
    Enums.cpp

    ADDITIONAL_HEADER_DIRS
    ${LLCOMPILER_INCLUDE_DIR}/llcompiler/Pipeline

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransforms
    MLIRLLHTransforms
    MLIRLLHToTosa
    MLIRTosaTransforms
    MLIRTosaToArith
    MLIRTosaToTensor
    MLIRTosaToLinalg
    MLIRTosaToSCF
    MLIRTosaExTransforms
    MLIRTensorToLinalg
    MLIRAffineToStandard
)

set_property(GLOBAL APPEND PROPERTY LLCOMPILER_COMMON_PIPELINE_DEPENDS MLIRFuncInlinerExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_COMMON_PIPELINE_DEPENDS MLIRTensorInferTypeOpInterfaceImpl)

llcompiler_install_mlir_library(LLCCommonPipeline PIPELINE)