# LLVM && MLIR

add_llvm("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
list(APPEND THIRD_PARTY_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/include"
    "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir/include"
    "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir/include")

# ONNX
add_onnx("${CMAKE_CURRENT_SOURCE_DIR}/onnx")
list(APPEND THIRD_PARTY_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx")

# spdlog
add_spdlog("${CMAKE_CURRENT_SOURCE_DIR}/spdlog")

set(THIRD_PARTY_INCLUDE_DIRS ${THIRD_PARTY_INCLUDE_DIRS} PARENT_SCOPE)
