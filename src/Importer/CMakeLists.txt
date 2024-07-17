add_mlir_library(LLCompilerImporter
    Importer.cpp
    LLHOpBuilder.cpp
    OnnxImport.cpp
    OpBuilder.cpp
    Utility.cpp

    DEPENDS
    LLCompilerSupport
    LLCLLHDialect

    LINK_COMPONENTS
    LLVMOption

    LINK_LIBS PUBLIC
    onnx
)
