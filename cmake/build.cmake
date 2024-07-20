# build and install lib
function(llcompiler_add_library name)
    cmake_parse_arguments(ARG
        "SHARED;DISABLE_INSTALL"
        ""
        "SRC_FILES;ADDITIONAL_INCLUDE_DIRS;DEPENDS;LINKS;DEFINITIONS;"
        ${ARGN}
    )

    if(ARG_ADDITIONAL_INCLUDE_DIRS)
        foreach(include_dir ${ARG_ADDITIONAL_INCLUDE_DIRS})
            include_directories(${include_dir})
        endforeach()
    endif()

    if(NOT ARG_SRC_FILES)
        set(LIBTYPE INTERFACE)
    else()
        if(ARG_SHARED)
            set(LIBTYPE SHARED)
        else()
            if(BUILD_SHARED_LIBS)
                set(LIBTYPE SHARED)
            else()
                set(LIBTYPE STATIC)
            endif()
        endif()
    endif()

    add_library(${name} ${LIBTYPE} ${ARG_SRC_FILES})
    message("add ${LIBTYPE} lib: ${name}.")

    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()

    if(ARG_LINKS)
        cmake_parse_arguments(LINKS_ARG
            ""
            ""
            "PUBLIC;PRIVATE;INTERFACE"
            ${ARG_LINKS})
        target_link_libraries(${name} PUBLIC ${LINKS_ARG_PUBLIC})
        target_link_libraries(${name} PRIVATE ${LINKS_ARG_PRIVATE})
        target_link_libraries(${name} INTERFACE ${LINKS_ARG_INTERFACE})
    endif()

    if(ARG_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${ARG_DEFINITIONS})
    endif()

    set_target_properties(${name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set_property(GLOBAL APPEND PROPERTY LLCOMPILER_ALL_TARGETS ${name})

    if(NOT DISABLE_INSTALL)
        llcompiler_install_library(${name})
    endif()
endfunction()

function(llcompiler_install_library name)
    install(TARGETS ${name}
        EXPORT ${name}Targets
        RUNTIME DESTINATION ${LLCOMPILER_INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${LLCOMPILER_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${LLCOMPILER_INSTALL_LIB_DIR})
    set_property(GLOBAL APPEND PROPERTY LLCOMPILER_INSTALLED_TARGETS ${name})
endfunction()

function(llcompiler_install_mlir_library name)
    set_target_properties(${name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set_property(GLOBAL APPEND PROPERTY LLCOMPILER_ALL_TARGETS ${name})
    set_property(GLOBAL APPEND PROPERTY LLCOMPILER_MLIR_TARGETS ${name})
    llcompiler_install_library(${name})
endfunction()