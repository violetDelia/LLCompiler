# #---------- compiler ----------##
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
llcompiler_add_executable(llcompiler
    "${CMAKE_CURRENT_SOURCE_DIR}/llcompiler.cpp"

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS
    LLCompiler
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIRExport
    MLIRTransforms
    onnx
)
