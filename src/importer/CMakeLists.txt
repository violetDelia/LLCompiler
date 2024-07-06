set(LLCompilerImporter_LINK_LIBS
    LLCompilerSupport
    onnx
)
llcompiler_add_library(LLCompilerImporter
    LLCompilerImporter_LINK_LIBS
    LLCOMPILER_CURRENT_DIR_FILES
)