CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# #---------- options cache----------##
unset(BUILD_SHARED_LIBS CACHE)
unset(BUILD_LLCOMPILER_TEST CACHE)
unset(STATIC_WINDOWS_RUNTIME CACHE)
unset(BUILD_LLCOMPILER_DOCS CACHE)
unset(BUILD_LLCOMPILER_LOG CACHE)
unset(CMAKE_EXPORT_COMPILE_COMMANDS CACHE)
unset(LLCOMPILER_BUILD_WARNINGS CACHE)

# TODO: discard for remove onnx
# unset(LLCOMPILER_BUILD_WITH_ONNX_ML CACHE)

# #---------- options ----------##
option(BUILD_SHARED_LIBS "build shared libs" ON)
option(BUILD_LLCOMPILER_TEST "build test" ON)
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)
option(BUILD_LLCOMPILER_DOCS "build examples" OFF)
option(BUILD_LLCOMPILER_LOG "build has log" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "export compiler_commands" ON)
option(LLCOMPILER_BUILD_WARNINGS "open wraning when build llcompiler" ON)

# option(LLCOMPILER_BUILD_WITH_ONNX_ML "if disable,will throw error when convert onnx model" OFF)

# #---------- cmake setting ----------##
project(LLCompiler LANGUAGES C CXX)
set(LLCOMPILER_VERSION 0.0.1)

# #---------- cmake config ----------##
file(GLOB_RECURSE LLCOMPILER_CMAKE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake)

foreach(CMAKE_FILE ${LLCOMPILER_CMAKE_FILES})
    include(${CMAKE_FILE})
endforeach()

# #---------- define absolute path ----------##
set(LLCOMPILER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LLCOMPILER_OPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ops")
set(LLCOMPILER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LLCOMPILER_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(LLCOMPILER_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools")
set(LLCOMPILER_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(LLCOMPILER_DOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs")
set(LLCOMPILER_BUILD_RUNTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(LLCOMPILER_BUILD_LIBRARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(LLCOMPILER_BUILD_ARCHIVE_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(LLCOMPILER_BUILD_TOOLS_DIR "${CMAKE_CURRENT_BINARY_DIR}/tools")

# #---------- define install absolute path ----------##
set(LLCOMPILER_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(LLCOMPILER_INSTALL_INCLUDE_DIR "${LLCOMPILER_INSTALL_DIR}/include")
set(LLCOMPILER_INSTALL_CMAKE_DIR "${LLCOMPILER_INSTALL_DIR}/lib/cmake/llcompiler")
set(LLCOMPILER_INSTALL_LIBRARY_DIR "${LLCOMPILER_INSTALL_DIR}/lib")
set(LLCOMPILER_INSTALL_RUNTIME_DIR "${LLCOMPILER_INSTALL_DIR}/bin")
set(LLCOMPILER_INSTALL_ARCHIVE_DIR "${LLCOMPILER_INSTALL_DIR}/lib")

# #---------- define  dirs ----------##
list(APPEND LLCOMPILER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND LLCOMPILER_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include")
list(APPEND LLCOMPILER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND LLCOMPILER_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/src")
list(APPEND SPDLOG_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include")
list(APPEND PYBIND11_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include")
list(APPEND STABLEHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stablehlo")
list(APPEND STABLEHLO_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/third_party/stablehlo")
list(APPEND LLVM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/include")
list(APPEND LLVM_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/include")
list(APPEND MLIR_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/mlir/include")
list(APPEND MLIR_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include")
list(APPEND SYMENGINE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/symengine_loc")
list(APPEND SYMENGINE_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/third_party/symengine_loc")
list(APPEND LLVM_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm")
list(APPEND LLVM_CMAKE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm")
list(APPEND MLIR_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake/mlir")
list(APPEND MLIR_CMAKE_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")

# #---------- external library ----------##
add_subdirectory(${LLCOMPILER_THIRDPARTY_DIR})

# #---------- sub library ----------##
if(BUILD_LLCOMPILER_LOG)
    add_definitions(-DLLCOMPILER_HAS_LOG)
endif()

include_directories(${LLCOMPILER_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_subdirectory(${LLCOMPILER_OPS_DIR})
add_subdirectory("${LLCOMPILER_INCLUDE_DIR}/llcompiler")
add_subdirectory(${LLCOMPILER_SOURCE_DIR})
add_subdirectory(${LLCOMPILER_TOOLS_DIR})

if(BUILD_LLCOMPILER_TEST)
    list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
    list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
    include(LLVMConfig)
    include(MLIRConfig)
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    add_subdirectory(test)
endif()

# #---------- install ----------##
include(CMakePackageConfigHelpers)

# # 安装头文件
install(DIRECTORY "${LLCOMPILER_INCLUDE_DIR}/" DESTINATION "${LLCOMPILER_INSTALL_INCLUDE_DIR}" FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.mlir"
    EXCLUDE
    PATTERN "LibraryConfig.h"
    PATTERN "LibraryConfig.install.h"
)
install(DIRECTORY "${LLCOMPILER_GENERATE_INCLUDE_DIR}/" DESTINATION "${LLCOMPILER_INSTALL_INCLUDE_DIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.inc.d"
)

# 生成setup.cfg文件
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
)

# 生成Config.cmake文件并安装
configure_package_config_file("${LLCOMPILER_CMAKE_DIR}/LLCompilerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LLCompilerConfig.cmake"
    INSTALL_DESTINATION ${LLCOMPILER_INSTALL_CMAKE_DIR}
)

# # 生成ConfigVersion.cmake文件
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/LLCompilerConfigVersion.cmake"
    VERSION ${LLCOMPILER_VERSION}
    COMPATIBILITY SameMajorVersion)

# # 安装LLFrameConfigVersion.cmake文件
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/LLCompilerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LLCompilerConfigVersion.cmake"
    DESTINATION ${LLCOMPILER_INSTALL_CMAKE_DIR})

# #---------- message ----------##
message(STATUS BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS})
message(STATUS STATIC_WINDOWS_RUNTIME: ${STATIC_WINDOWS_RUNTIME})
message(STATUS CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS})
message(STATUS BUILD_LLCOMPILER_LOG: ${BUILD_LLCOMPILER_LOG})
message(STATUS CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX})

# #---------- this is something for quick test ----------##
llcompiler_add_executable(main
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)
target_include_directories(main PRIVATE ${SYMENGINE_INCLUDE_DIRS})
target_link_libraries(main PUBLIC symengine)