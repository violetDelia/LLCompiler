# #---------- compiler ----------##
llcompiler_add_executable(onnx-to-mlir
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx-to-mlir.cpp"

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS
    LLCompiler
    onnx
)
