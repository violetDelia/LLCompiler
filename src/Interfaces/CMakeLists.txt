
llcompiler_add_library(MLIRSymbolShapeOpInterfaces
    SRC_FILES
    SymbolShapeOpInterfaces.cpp

    DEPENDS
    MLIRSymbolShapeOpInterfacesIncGen

    LINKS PUBLIC
    MLIRIR
)

llcompiler_add_library(MLIRBraodcastableOpInterfaces
    SRC_FILES
    BraodcastableOpInterfaces.cpp

    DEPENDS
    MLIRBraodcastableOpInterfacesIncGen

    LINKS PUBLIC
    MLIRIR
)
