get_property(pipeline_depends GLOBAL PROPERTY LLCOMPILER_PIPELINE_DEPENDS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

llcompiler_add_library(LLCompiler
    SRC_FILES
    Init.cpp
    Compiler.cpp
    Tensor.cpp
    CompileOptions.cpp
    Command.cpp
    Execution.cpp

    ADDITIONAL_INCLUDE_DIRS

    LINKS PUBLIC
    MLIRIR
    LLCSupport
    LLCPipelines
    MLIRExecutionEngine
    LLVMX86AsmParser
    ${pipeline_depends}
)
