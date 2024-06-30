set(LLCOMPILER_CXX_FLAGS ${CMAKE_CXX_FLAGS})
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4996>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd5054>)
endif()
if(STATIC_WINDOWS_RUNTIME)
    option(ONNX_USE_MSVC_STATIC_RUNTIME "static" ON)
endif()

unset(BENCHMARK_ENABLE_EXCEPTIONS CACHE)
unset(BENCHMARK_ENABLE_TESTING CACHE)
unset(LLVM_BUILD_EXAMPLES CACHE)
unset(LLVM_ENABLE_ASSERTIONS CACHE)
unset(LLVM_ENABLE_ASSERTIONS CACHE)
set(LLVM_TARGETS_TO_BUILD "Native;NVPTX;AMDGPU;RISCV"
    CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING
    "Semicolon-separated list of projects to build (${LLVM_KNOWN_PROJECTS}), or \"all\".")
option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)
option(BENCHMARK_ENABLE_EXCEPTIONS "Enable the use of exceptions in the benchmark library." OFF)
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
option(LLVM_BUILD_EXAMPLES "Build the LLVM example programs. If OFF, just generate build targets." ON)
option(LLVM_ENABLE_ASSERTIONS "Enable assertions" ON)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
set(LLVM_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/include" PARENT_SCOPE)
set(LLVM_GENERATE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/include" PARENT_SCOPE)
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/pybind11")

#-------------------------------------------------------------------------------
# Configure out-of-tree vs in-tree build
#-------------------------------------------------------------------------------
# set(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/inlcude)
# set(LLVM_PROJECT_SOURCE_DIR ${LLVM_MAIN_SRC_DIR}/..)
# message(STATUS "Using LLVM Project ${LLVM_PROJECT_SOURCE_DIR}")

# set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
# set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
# set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
# set(LLVM_MLIR_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
# set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
# # Add MLIR and LLVM headers to the include path
# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})

# # Configure CMake.
# list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
# list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

# find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

# include(TableGen)
# include(AddLLVM)
# include(AddMLIR)
# include(HandleLLVMOptions)
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/onnx-mlir")
cpmpile_spdlog("${CMAKE_CURRENT_SOURCE_DIR}/spdlog")