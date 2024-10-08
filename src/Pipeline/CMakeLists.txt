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
    MLIRLLVMExtensionTransforms
    MLIRIndexExtensionTransforms
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
    MLIRConvertToLLVMPass
    MLIRLLVMIRTransforms
    
    
)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRFuncInlinerExtension)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRTensorInferTypeOpInterfaceImpl)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRUBToLLVM)
set_property(GLOBAL APPEND PROPERTY LLCOMPILER_PIPELINE_DEPENDS MLIRMemRefToLLVM)
