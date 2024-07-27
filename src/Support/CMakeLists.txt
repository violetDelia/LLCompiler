llcompiler_add_library(LLCSupport
    SRC_FILES
    Logger.cpp
    Option.cpp

    LINKS PUBLIC
    LLVMOption

    PRIVATE
    spdlog
)
