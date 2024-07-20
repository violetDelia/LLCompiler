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

# add_library(LLCompilerImporter Importer.cpp
# LLHOpBuilder.cpp
# OnnxImport.cpp
# OpBuilder.cpp
# Utility.cpp)

# target_link_libraries(LLCompilerImporter PUBLIC LLCompilerSupport LLCLLHDialect LLVMOption onnx)

# if(LLCOMPILER_BUILD_WITH_ONNX_ML)
# target_compile_definitions(LLCompilerImporter PRIVATE ONNX_ML)
# endif()