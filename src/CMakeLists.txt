add_subdirectory(Support)
add_subdirectory(Importer)
add_subdirectory(Dialect)
add_subdirectory(Compiler)

llcompiler_add_library(LLCompilerAll
    SRC_FILES

    DEPENDS
    LLCompilerSupport
    LLCLLHDialect
    LLCLLHPasses
    LLCompilerImporter
    LLCompiler

    LINKS
)
