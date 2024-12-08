//    Copyright 2024 时光丶人爱

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
#ifndef SYMBOLINFER_PASS
#define SYMBOLINFER_PASS

include "mlir/Pass/PassBase.td"

def InferSymbolShapePass : Pass<"infer-symbol-shape","ModuleOp"> {
  let summary = "generate symbol";
  let description = [{
      形状推导并生成符号信息;
      ```mlir
      %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
      ```
      ====>
      ```mlir
      "llh.symbolic_int"() <{sym_name = "s3"}> : () -> ()
      "llh.symbolic_int"() <{sym_name = "s2"}> : () -> ()
      "llh.symbolic_int"() <{sym_name = "s1"}> : () -> ()
      "llh.symbolic_int"() <{sym_name = "s0"}> : () -> ()
      %0 = "llh.add"(%arg0, %arg0) : (tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>, tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>) -> tensor<?x?x?x?xf32, #llh.encoding<shapes = @s0, @s1, @s2, @s3>>
      ```
  }];
  let dependentDialects = [ "mlir::llh::LLHDialect",
                            "::mlir::arith::ArithDialect",
                            "::mlir::tensor::TensorDialect",
                            "::mlir::memref::MemRefDialect"];
  let options = [
    Option<"CleanSymbolCache", "clean-symbol-cache", "bool", /*default=*/"true",
           "测试用,防止多个相同地址的module.">,
    Option<"UseEncoding", "use-encoding", "bool", /*default=*/"true",
           "use llh.encoding.">,
  ];
}

def SymbolFoldPass : Pass<"symbol-fold","ModuleOp"> {
  let summary = "symbol fold";
  let description = [{
  }];
  let dependentDialects = [ "mlir::llh::LLHDialect",
                            "::mlir::arith::ArithDialect",
                            "::mlir::tensor::TensorDialect",
                            "::mlir::memref::MemRefDialect"];
}

def UnloadAndBindEncoding : Pass<"unload-and-bind-encoding","ModuleOp">{
  let summary = "卸载tensor上的EncodingAttr到EncodingBindOp代替";
  let description = [{
    1.防止与其他第三方库和标准MLIR不兼容。
  }];
}

def SinkBindEncoding : Pass<"sink-bind-encoding","ModuleOp">{
  let summary = "Bufferize的预处理";
  let description = [{
    
  }];

}

def RemoveSymbolPass : Pass<"remove-symbol","ModuleOp">{
  let summary = "去除符号";
  let description = [{
    
  }];
}
#endif // SYMBOLINFER_PASS