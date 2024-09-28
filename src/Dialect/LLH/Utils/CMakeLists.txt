llcompiler_add_library(MLIRLLHUtils
  SRC_FILES
  Utils.cpp
  SymbolAnalysis.cpp

  DEPENDS
  MLIRLLHDialectIncGen
  

  LINKS PUBLIC
  MLIRLLHDialect
)