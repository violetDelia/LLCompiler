CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
file(GLOB_RECURSE LLCOMPILER_CMAKE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake)

foreach(CMAKE_FILE ${LLCOMPILER_CMAKE_FILES})
    include(${CMAKE_FILE})
endforeach()

# #---------- options cache ----------##
unset(BUILD_SHARED_LIBS CACHE)
unset(TEST CACHE)
unset(STATIC_WINDOWS_RUNTIME CACHE)
unset(BUILD_LLCOMPILER_LOG CACHE)
unset(CMAKE_EXPORT_COMPILE_COMMANDS CACHE)
unset(BUILD_NO_WARINGS CACHE)

# #---------- options ----------##
# cant set to OFF
option(BUILD_SHARED_LIBS "build shared libs" OFF)
option(TEST "run test" ON)
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
option(BUILD_LLCOMPILER_LOG "build has log" ON)
option(CMAKE_EXPORT_COMPILE_COMMANDS "export compiler_commands" ON)
option(BUILD_NO_WARINGS "remove warings of compiler" ON)

# #---------- cmake setting ----------##
project(LLCompiler LANGUAGES C CXX)
set(LLCOMPILER_VERSION 1.0)

# #---------- compiler setting ----------##
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_DEBUG_POSTFIX d)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

if(MSVC)
    if(STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

if(BUILD_NO_WARINGS)
    if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Wall>)
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-w>)
    endif()
endif()

# #---------- define absolute path ----------##
cmake_path(APPEND LLCOMPILER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} "include")
cmake_path(APPEND LLCOMPILER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} "src")
cmake_path(APPEND LLCOMPILER_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR} "cmake")
cmake_path(APPEND LLCOMPILER_TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR} "tools")
cmake_path(APPEND LLCOMPILER_THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR} "third_party")
cmake_path(APPEND LLCOMPILER_DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR} "docs")

# #---------- define install absolute path ----------##
cmake_path(APPEND LLCOMPILER_INSTALL_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "install")
cmake_path(APPEND LLCOMPILER_INSTALL_INCLUDE_DIR ${INSTALL_DIR} "include")
cmake_path(APPEND LLCOMPILER_INSTALL_LIB_DIR ${INSTALL_DIR} "lib")
cmake_path(APPEND LLCOMPILER_INSTALL_RUNTIME_DIR ${INSTALL_DIR} "bin")
cmake_path(APPEND LLCOMPILER_INSTALL_CMAKE_DIR ${INSTALL_DIR} "cmake")

# #---------- external library ----------##
# msvc编译器设置
add_subdirectory(${LLCOMPILER_THIRDPARTY_DIR})

# set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LLVM_MAIN_INCLUDE_DIR})
include_directories(${LLVM_GENERATE_INCLUDE_DIR})
include_directories(${MLIR_MAIN_INCLUDE_DIR})
include_directories(${MLIR_GENERATE_INCLUDE_DIR})

# #---------- sub library ----------##
if(BUILD_LLCOMPILER_LOG)
    add_definitions(-DLLCOMPILER_HAS_LOG)
endif()

set(LLCOMPILER_CURRENT_DIR_FILES "llcompiler_current_dir_fils")
include_directories(${LLCOMPILER_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory("${LLCOMPILER_INCLUDE_DIR}/llcompiler")
add_subdirectory(${LLCOMPILER_SOURCE_DIR})
add_subdirectory(${LLCOMPILER_TOOLS_DIR})

# #---------- main ----------##
add_executable(main "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_include_directories(main PRIVATE
    $<BUILD_INTERFACE:${LLCOMPILER_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_DIR}>
)
target_link_libraries(main PRIVATE LLCompilerAll
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)

# #---------- message ----------##
message(STATUS BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS})
message(STATUS STATIC_WINDOWS_RUNTIME: ${STATIC_WINDOWS_RUNTIME})
message(STATUS CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS})
message(STATUS BUILD_LLCOMPILER_LOG: ${BUILD_LLCOMPILER_LOG})
message(STATUS CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS})
message(STATUS CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX})
