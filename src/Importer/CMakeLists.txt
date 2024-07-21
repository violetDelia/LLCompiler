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

    DEPENDS
    LLVMOption
    LLCompilerSupport
    LLCLLHDialect

    LINKS
    PRIVATE
    onnx
)
