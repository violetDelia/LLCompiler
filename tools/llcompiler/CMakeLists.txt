# #---------- compiler ----------##
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
llcompiler_add_executable(llcompiler
    "${CMAKE_CURRENT_SOURCE_DIR}/llcompiler.cpp"

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS
    LLVMSupport
    LLCompiler
    MLIRPass
    MLIRLLHToTosa
    MLIROptLib
)
