
pybind11_add_module(example "${CMAKE_CURRENT_SOURCE_DIR}/example.cpp")
install(TARGETS example DESTINATION ${CMAKE_CURRENT_BINARY_DIR})