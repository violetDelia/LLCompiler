llcompiler_add_library(LLCLLHUtils
  SRC_FILES
  Utils.cpp
  Broadcast.cpp
  InferSymbol.cpp
  SymbolAnalysis.cpp

  ADDITIONAL_INCLUDE_DIRS
  ${SYMENGINE_INCLUDE_DIRS}

  DEPENDS
  LLCLLHDialectIncGen
  

  LINKS PUBLIC
  LLCLLHDialect
)