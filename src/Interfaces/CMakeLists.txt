
add_mlir_library(MLIRSymbolShapeOpInterfaces
    SymbolShapeOpInterfaces.cpp

    DEPENDS
    MLIRSymbolShapeOpInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
