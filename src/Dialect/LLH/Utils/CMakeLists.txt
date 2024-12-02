llcompiler_add_library(LLCLLHUtils
  SRC_FILES
  Utils.cpp
  SymbolAnalysis.cpp
  InferSymbol.cpp
  Broadcast.cpp

  ADDITIONAL_INCLUDE_DIRS
  ${SYMENGINE_INCLUDE_DIRS}

  DEPENDS
  LLCLLHDialectIncGen
  

  LINKS PUBLIC
  MLIRLLHDialect
)