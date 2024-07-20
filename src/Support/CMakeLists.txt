llcompiler_add_library(LLCompilerSupport
    SRC_FILES
    Core.cpp
    Logger.cpp
    Option.cpp

    DEPENDS
    LLVMOption

    LINKS PRIVATE
    spdlog
)
