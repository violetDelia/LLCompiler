set(LLCompilerSupport_LINK_LIBS
    LLVMOption
    spdlog
)
llcompiler_add_library(LLCompilerSupport
    LLCompilerSupport_LINK_LIBS
    LLCOMPILER_CURRENT_DIR_FILES
)
