llcompiler_add_library(LLCCommonPipeline
    SRC_FILES
    CommonPipeline.cpp
    Enums.cpp

    DEPENDS
    LLCConversionPassIncGen
    MLIRLLHToTosaIncGen

    LINKS PUBLIC
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
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRControlFlowToSPIRV
    MLIRSPIRVToLLVM
    MLIRSCFToSPIRV
    MLIRFuncToLLVM
    MLIRLLVMIRTransforms
)

# add_mlir_dialect_library(LLCCommonPipeline
# CommonPipeline.cpp
# Enums.cpp

# ADDITIONAL_HEADER_DIRS
# ${LLCOMPILER_INCLUDE_DIR}/llcompiler/Pipeline

# LINK_LIBS PUBLIC
# MLIRPass
# MLIRTransforms
# MLIRLLHTransforms
# MLIRLLHToTosa
# MLIRTosaTransforms
# MLIRTosaToArith
# MLIRTosaToTensor
# MLIRTosaToLinalg
# MLIRTosaToSCF
# MLIRTosaExTransforms
# MLIRTensorToLinalg
# MLIRAffineToStandard
# MLIRSCFToControlFlow
# MLIRControlFlowToLLVM
# MLIRMemRefToLLVM
# MLIRReconcileUnrealizedCasts
# MLIRControlFlowToSPIRV
# MLIRSPIRVToLLVM
# MLIRSCFToSPIRV
# MLIRFuncToLLVM
# MLIRLLVMIRTransforms
# )
# llcompiler_install_mlir_target(LLCCommonPipeline PIPELINE)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_COMMON_PIPELINE_DEPENDS MLIRFuncInlinerExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_COMMON_PIPELINE_DEPENDS MLIRTensorInferTypeOpInterfaceImpl)
