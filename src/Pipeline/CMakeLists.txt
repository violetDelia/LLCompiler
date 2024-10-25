llcompiler_add_library(LLCPipelines
    SRC_FILES
    CommonPipeline.cpp
    BasicPipeline.cpp

    DEPENDS
    LLCConversionPassIncGen
    MLIRLLHToTosaIncGen

    LINKS PUBLIC
    MLIRPass
    MLIRTransforms
    MLIRLLHUtils
    MLIRLLHTransforms
    MLIRLLHToTosa
    MLIRLLHToArith
    MLIRLLHToTensor
    MLIRLLHToHLO
    MLIRLLVMExtensionTransforms
    MLIRIndexExtensionTransforms
    MLIRTosaTransforms
    MLIRTosaToLinalgExtension
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
    MLIRConvertToLLVMPass
    MLIRLLVMIRTransforms
    MLIRConvertToSPIRVPass
    MLIRTensorToSPIRV
    MLIRControlFlowToSCF
    MLIRBufferizationToMemRef
    MLIRIndexToLLVM
    MhloDialect
    MhloPasses
    MhloToMemrefConversion
    MhloToStandard
    MhloToLinalg
    DeallocationPasses
    MLIRBufferTransforms
    StablehloPasses
    StablehloTOSATransforms
    StablehloLinalgTransforms
)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRFuncInlinerExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTensorInferTypeOpInterfaceImpl)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRUBToLLVM)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRMemRefToLLVM)
