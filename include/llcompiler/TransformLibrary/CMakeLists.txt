set(LLCOMPILER_TANSFORMLIB_INSTALL_DIR "${LLCOMPILER_INSTALL_INCLUDE_DIR}/TransformLibrary")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.h"
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.install.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.install.h"
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LibraryEntry.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/LibraryEntry.h"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LibraryPath.install.h"
    DESTINATION ${LLCOMPILER_TANSFORMLIB_INSTALL_DIR} RENAME LibraryPath.h)