add_subdirectory(python)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LLCOMPILER_TEST_DEPENDS
  FileCheck count not
  llc-opt
  mlir-runner
  mlir-opt
  mlir_c_runner_utils
  mlir_runner_utils
)

add_lit_testsuite(check-llcompiler "Running the llcompiler regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LLCOMPILER_TEST_DEPENDS}
)
set_target_properties(check-llcompiler PROPERTIES FOLDER "Test")

