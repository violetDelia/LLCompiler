get_property(pipeline_depends GLOBAL PROPERTY LLCOMPILER_PIPELINE_DEPENDS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

llcompiler_add_library(LLCompiler 
    SRC_FILES
    Init.cpp
    Entrance.cpp
    Engine.cpp
    Tensor.cpp
    #Utility.cpp

    ADDITIONAL_INCLUDE_DIRS
    # ${ONNX_INCLUDE_DIR}
    # ${ONNX_INCLUDE_GENERATE_DIR}
    # ${PROTOBUF_INCLUDE_DIR}
    # ${ABSL_INCLUDE_DIR}

    LINKS PUBLIC
    MLIRIR
    LLCSupport
    LLCFrontendCore
    LLCMLIRUtility
    LLCPipelines
    MLIRExecutionEngine
    LLVMX86AsmParser
    ${pipeline_depends}


)
# Note: discard 
# if(LLCOMPILER_BUILD_WITH_ONNX_ML)
#     target_compile_definitions(LLCompiler PRIVATE ONNX_ML)
# endif()