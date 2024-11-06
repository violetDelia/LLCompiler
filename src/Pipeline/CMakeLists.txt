llcompiler_add_library(LLCPipelines
    SRC_FILES
    CommonPipeline.cpp
    BasicPipeline.cpp
    TransformPipeline.cpp

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
    MLIRLLHToShape
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
    MhloToStablehlo
)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRFuncInlinerExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTensorInferTypeOpInterfaceImpl)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRUBToLLVM)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRMemRefToLLVM)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRGPUTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRFuncTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRLinalgTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRMemRefTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRSCFTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTensorTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRVectorTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTransformDebugExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTransformDialectIRDLExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTransformLoopExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTransformPDLExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRAffineTransformOps)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRBufferizationTransformOps)