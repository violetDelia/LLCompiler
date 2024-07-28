llcompiler_add_library(LLCFrontendOnnx
    SRC_FILES
    OnnxImport.cpp
    OnnxBuilder.cpp

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS PUBLIC
    LLCSupport
    LLCFrontendCore

    PRIVATE
    onnx
)

if(LLCOMPILER_BUILD_WITH_ONNX_ML)
    target_compile_definitions(LLCFrontendOnnx PRIVATE ONNX_ML)
endif()
