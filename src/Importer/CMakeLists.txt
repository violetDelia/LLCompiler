llcompiler_add_library(LLCompilerImporter
    SRC_FILES
    Importer.cpp
    OnnxImport.cpp
    OpBuilder.cpp
    Utility.cpp

    DEPENDS
    LLCompilerSupport
    LLCLLHDialect
    
    LINKS
    onnx
    
)
