set(LLCOMPILER_TANSFORMLIB_INSTALL_DIR "${LLCOMPILER_INSTALL_INCLUDE_DIR}/llcompiler/TransformLibrary")

if(DEVELOPER_MODE)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/LibraryPath.h"
    )
else()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.install.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/LibraryPath.h"
    )
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryEntry.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LibraryEntry.h"
)
