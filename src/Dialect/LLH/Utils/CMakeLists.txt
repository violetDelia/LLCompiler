llcompiler_add_library(LLCLLHUtils
  SRC_FILES
  Utils.cpp
  SymbolAnalysis.cpp
  InferSymbol.cpp
  Broadcast.cpp

  DEPENDS
  MLIRLLHDialectIncGen
  

  LINKS PUBLIC
  MLIRLLHDialect
)