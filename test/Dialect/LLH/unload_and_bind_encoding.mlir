// RUN: llc-opt --split-input-file --unload-and-bind-encoding %s| FileCheck %s

// CHECK-LABEL: None
func.func @None(%arg0: i64, %arg1: i64, %arg2: tensor<?x?x224x224xf32>) ->(){
    %1 = "llh.aot"(%arg0) <{name = "aot1"}> : (i64) -> i64
    %2 = "llh.aot"(%arg1,%arg2) <{name = "aot2"}> : (i64,tensor<?x?x224x224xf32>) -> tensor<?x?x224x224xf32>
    %3 = "llh.aot"() <{name = "aot3"}> : () -> i64
    %4 = "llh.aot"() <{name = "aot4"}> : () -> i64
    %5 = "llh.aot"() <{name = "aot5"}> : () -> i64
    return
}



