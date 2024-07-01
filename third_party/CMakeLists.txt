

# LLVM && MLIR
unset(BENCHMARK_ENABLE_EXCEPTIONS CACHE)
unset(BENCHMARK_ENABLE_TESTING CACHE)
unset(LLVM_BUILD_EXAMPLES CACHE)
unset(LLVM_ENABLE_ASSERTIONS CACHE)
unset(LLVM_ENABLE_ASSERTIONS CACHE)
unset(LLVM_INCLUDE_TESTS CACHE)
set(LLVM_TARGETS_TO_BUILD "Native;NVPTX;AMDGPU;RISCV"
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING
    "Semicolon-separated list of projects to build (${LLVM_KNOWN_PROJECTS}), or \"all\".")
option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
option(BENCHMARK_ENABLE_EXCEPTIONS "Enable the use of exceptions in the benchmark library." OFF)
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
option(LLVM_BUILD_EXAMPLES "Build the LLVM example programs. If OFF, just generate build targets." ON)
option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
option(LLVM_INCLUDE_TESTS "Generate build targets for the LLVM unit tests." OFF)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
set(LLVM_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/include" PARENT_SCOPE)
set(LLVM_GENERATE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/include" PARENT_SCOPE)


# ONNX
if(STATIC_WINDOWS_RUNTIME)
    option(ONNX_USE_MSVC_STATIC_RUNTIME "static" ON)
endif()
unset(ABSL_ENABLE_INSTALL CACHE)
option(ABSL_ENABLE_INSTALL "" OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/onnx")

cpmpile_spdlog("${CMAKE_CURRENT_SOURCE_DIR}/spdlog")