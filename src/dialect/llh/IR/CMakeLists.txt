add_mlir_dialect_library(LLCLLHIR
    llh_dialect.cpp
    llh_ops.cpp

    DEPENDS
    MLIRLLHOpsIncGen

    LINK_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
)
