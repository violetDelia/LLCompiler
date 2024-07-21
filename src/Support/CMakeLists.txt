llcompiler_add_library(LLCompilerSupport
    SRC_FILES
    Core.cpp
    Logger.cpp
    Option.cpp

    LINKS
    PUBLIC
    LLVMOption
    PRIVATE
    spdlog
)
