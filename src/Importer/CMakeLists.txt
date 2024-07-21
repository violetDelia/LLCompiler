llcompiler_add_library(LLCompilerImporter
    SRC_FILES
    Importer.cpp
    LLHOpBuilder.cpp
    OnnxImport.cpp
    OpBuilder.cpp
    Utility.cpp

    ADDITIONAL_INCLUDE_DIRS
    ${ONNX_INCLUDE_DIR}
    ${ONNX_INCLUDE_GENERATE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${ABSL_INCLUDE_DIR}

    LINKS
    PUBLIC
    LLCompilerSupport
    LLCLLHDialect
    PRIVATE
    onnx
    LLVMOption
    LLCMLIRUtility
)

if(LLCOMPILER_BUILD_WITH_ONNX_ML)
    target_compile_definitions(LLCompilerImporter PRIVATE ONNX_ML)
endif()
