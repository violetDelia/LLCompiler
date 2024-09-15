llcompiler_add_library(LLCompiler
    SRC_FILES
    Init.cpp
    #Utility.cpp

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS PUBLIC
    MLIRIR
    LLCSupport

)

if(LLCOMPILER_BUILD_WITH_ONNX_ML)
    target_compile_definitions(LLCompiler PRIVATE ONNX_ML)
endif()