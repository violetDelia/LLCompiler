# #---------- compiler ----------##
llcompiler_add_executable(compiler
    "${CMAKE_CURRENT_SOURCE_DIR}/compiler.cpp"

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS
    LLVMSupport
    LLCompiler
    onnx
)
