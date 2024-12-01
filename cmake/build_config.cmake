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

