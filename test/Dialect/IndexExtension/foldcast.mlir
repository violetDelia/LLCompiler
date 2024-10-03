// RUN: llc-opt --split-input-file --fold-index-cast %s| FileCheck %s

func.func @cast() ->(index) attributes {entrance}{
    %c3 = arith.constant 3 : index
    %13 = index.castu %c3 : index to i64
    %17 = index.castu %13 : i64 to index
    // CHECK: arith.constant
    // CHECK-NOT : index.castu
    return %17 : index
}