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

# #---------- compiler setting ----------##
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

if(MSVC)
    if(STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

if(LLCOMPILER_BUILD_WARNINGS)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Wall>)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w>)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-w>)
    endif()
endif()

add_compile_options(-fPIC)

# add_compile_options(-fno-rtti)

# #---------- define absolute path ----------##
set(LLCOMPILER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LLCOMPILER_OPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ops")
set(LLCOMPILER_GENERATE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
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

# set(CMAKE_INSTALL_RPATH ${LLCOMPILER_INSTALL_RUNTIME_DIR})
# #---------- define include dirs ----------##
set(LLCOMPILER_INCLUDE_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
set(SPDLOG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include")
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include")
set(STABLEHLO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stablehlo")
set(STABLEHLO_INCLUDE_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/stablehlo")
set(MHLO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlir-hlo")
set(MHLO_INCLUDE_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/mlir-hlo")
# set(ONNX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx")
# set(ONNX_INCLUDE_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/onnx")
# set(PROTOBUF_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/protobuf-src/src")
# set(ABSL_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/abseil-src")
set(LLVM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/llvm/include")
set(LLVM_INCLDE_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/include")
set(MLIR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llvm-project/mlir/include")
set(MLIR_INCLUDE_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include")

# #---------- external library ----------##
add_subdirectory(${LLCOMPILER_THIRDPARTY_DIR})

# #---------- sub library ----------##
if(BUILD_LLCOMPILER_LOG)
    add_definitions(-DLLCOMPILER_HAS_LOG)
endif()

include_directories(${LLCOMPILER_INCLUDE_DIR})
include_directories(${LLCOMPILER_INCLUDE_GENERATE_DIR})
include_directories(${LLVM_INCLUDE_DIR})
include_directories(${LLVM_INCLDE_GENERATE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_GENERATE_DIR})
include_directories(${SPDLOG_INCLUDE_DIR})
include_directories(${STABLEHLO_INCLUDE_DIR})
include_directories(${STABLEHLO_INCLUDE_GENERATE_DIR})
include_directories(${MHLO_INCLUDE_DIR})
include_directories(${MHLO_INCLUDE_GENERATE_DIR})

add_subdirectory(${LLCOMPILER_OPS_DIR})
add_subdirectory("${LLCOMPILER_INCLUDE_DIR}/llcompiler")
add_subdirectory(${LLCOMPILER_SOURCE_DIR})
add_subdirectory(${LLCOMPILER_TOOLS_DIR})

if(BUILD_LLCOMPILER_TEST)
    set(LLVM_CMAKE_DIR "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm")
    set(MLIR_CMAKE_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
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
install(DIRECTORY "${LLCOMPILER_INCLUDE_DIR}/" DESTINATION "${LLCOMPILER_INSTALL_INCLUDE_DIR}")
install(DIRECTORY "${LLCOMPILER_GENERATE_INCLUDE_DIR}/" DESTINATION "${LLCOMPILER_INSTALL_INCLUDE_DIR}")

# 生成setup.cfg文件
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg"
    INSTALL_DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
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