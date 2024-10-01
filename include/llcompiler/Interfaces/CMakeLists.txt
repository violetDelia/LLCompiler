

set(LLVM_TARGET_DEFINITIONS SymbolShapeOpInterfaces.td)
mlir_tablegen(SymbolShapeOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SymbolShapeOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRSymbolShapeOpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRSymbolShapeOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS BraodcastableOpInterfaces.td)
mlir_tablegen(BraodcastableOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(BraodcastableOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRBraodcastableOpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRBraodcastableOpInterfacesIncGen)
