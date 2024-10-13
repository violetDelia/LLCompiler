configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LLCOMPILER_TEST_DEPENDS
  FileCheck count not
  llc-opt
)

add_lit_testsuite(check-llc-python "Running the llcompiler regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLCOMPILER_TEST_DEPENDS}
)
set_target_properties(check-llc-python PROPERTIES FOLDER "Test")

