# #---------- compiler ----------##
add_executable(compiler "${CMAKE_CURRENT_SOURCE_DIR}/compiler.cpp")
target_include_directories(compiler PRIVATE
    $<BUILD_INTERFACE:${ONNX_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ONNX_INCLUDE_GENERATE_DIR}>
    $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${ABSL_INCLUDE_DIR}>
)

target_link_libraries(compiler
    PRIVATE
    LLCompilerImporter
    LLCompiler
    MLIRLLHDialect
    onnx
)