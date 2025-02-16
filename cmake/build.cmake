function(llcompiler_add_executable name src)
    cmake_parse_arguments(ARG
        "DISABLE_INSTALL"
        ""
        "ADDITIONAL_INCLUDE_DIRS;DEPENDS;LINKS;DEFINITIONS;"
        ${ARGN}
    )

    add_executable(${name} ${src})

    if(ARG_ADDITIONAL_INCLUDE_DIRS)
        target_include_directories(${name}
            PRIVATE
            $<BUILD_INTERFACE:${ARG_ADDITIONAL_INCLUDE_DIRS}>
            $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_INCLUDE_DIR}>)
    endif()

    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()

    if(ARG_LINKS)
        target_link_libraries(${name} PRIVATE ${ARG_LINKS})
    endif()

    if(ARG_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${ARG_DEFINITIONS})
    endif()

    if(NOT DISABLE_INSTALL)
        llcompiler_install_target(${name})
    endif()

    message("add executable: ${name}.")
endfunction()

function(llcompiler_add_library name)
    cmake_parse_arguments(ARG
        "SHARED;DISABLE_INSTALL"
        ""
        "SRC_FILES;ADDITIONAL_INCLUDE_DIRS;DEPENDS;LINKS;DEFINITIONS;"
        ${ARGN}
    )

    if(ARG_ADDITIONAL_INCLUDE_DIRS)
        include_directories(${ARG_ADDITIONAL_INCLUDE_DIRS})
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

    set_property(GLOBAL APPEND PROPERTY LLCOMPILER_ALL_TARGETS ${name})

    if(NOT ARG_DISABLE_INSTALL)
        llcompiler_install_target(${name})
    endif()
endfunction()

function(llcompiler_install_target name)
    set_target_properties(${name}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${LLCOMPILER_BUILD_ARCHIVE_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${LLCOMPILER_BUILD_LIBRARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${LLCOMPILER_BUILD_RUNTIME_DIR}
        INSTALL_RPATH ${LLCOMPILER_INSTALL_LIBRARY_DIR}
    )

    install(TARGETS ${name}
        EXPORT ${name}Targets
        RUNTIME DESTINATION ${LLCOMPILER_INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${LLCOMPILER_INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${LLCOMPILER_INSTALL_ARCHIVE_DIR}
        PUBLIC_HEADER DESTINATION ${LLCOMPILER_INSTALL_DIR}/include)
    set_property(GLOBAL APPEND PROPERTY LLCOMPILER_INSTALLED_TARGETS ${name})
endfunction()
