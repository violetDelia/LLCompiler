pybind11_add_module(entrance "${CMAKE_CURRENT_SOURCE_DIR}/Entrance.cpp")
target_link_libraries(entrance PUBLIC LLCompiler)
llcompiler_install_target(entrance)
