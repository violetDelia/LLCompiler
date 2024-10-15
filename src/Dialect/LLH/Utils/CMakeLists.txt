llcompiler_add_library(MLIRLLHUtils
  SRC_FILES
  Utils.cpp
  SymbolAnalysis.cpp
  InferSymbol.cpp

  DEPENDS
  MLIRLLHDialectIncGen
  

  LINKS PUBLIC
  MLIRLLHDialect
)