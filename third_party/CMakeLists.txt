# LLVM && MLIR

add_llvm("${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm")
set(LLVM_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/llvm/include" PARENT_SCOPE)
set(LLVM_GENERATE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/include" PARENT_SCOPE)
set(MLIR_MAIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project/mlir/include" PARENT_SCOPE)
set(MLIR_GENERATE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/llvm-project/llvm/tools/mlir/include" PARENT_SCOPE)

# ONNX
add_onnx("${CMAKE_CURRENT_SOURCE_DIR}/onnx")

# spdlog
add_spdlog("${CMAKE_CURRENT_SOURCE_DIR}/spdlog")
