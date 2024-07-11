function(llcompiler_add_library name)
    cmake_parse_arguments(ARG
        ""
        ""
        "SRC_FILES;DEPENDS;LINKS"
        ${ARGN}
    )

    # #---------- compile lib ----------##
    if(ARG_SRC_FILES)
        if(BUILD_SHARED_LIBS)
            add_library(${name} SHARED ${ARG_SRC_FILES})
        else()
            add_library(${name} STATIC ${ARG_SRC_FILES})
        endif()

        target_include_directories(${name}
            PRIVATE
            $<BUILD_INTERFACE:${LLCOMPILER_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_DIR}>
        )

    # #---------- INTERFACE lib ----------##
    else()
        add_library(${name} INTERFACE)
        target_include_directories(${name}
            INTERFACE
            $<BUILD_INTERFACE:${LLCOMPILER_INCLUDE_DIR}>
            $<INSTALL_INTERFACE:${LLCOMPILER_INSTALL_DIR}>
        )
    endif()

    set_target_properties(${name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()

    target_link_libraries(${name} PUBLIC ${ARG_LINKS})
endfunction()
