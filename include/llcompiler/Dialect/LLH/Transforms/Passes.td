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
#ifndef LLH_PASS
#define LLH_PASS

include "mlir/Pass/PassBase.td"

def TransformLayoutPass : Pass<"transform-layout","ModuleOp"> {
  let summary = "trans some llh op layout to nhwc for lowing to tosa";
  let description = [{
  }];
  let dependentDialects = [];
  let options = [
    Option<"TargetLayout", "target-layout", "::mlir::llh::Layout", /*default=*/"::mlir::llh::Layout::NHWC",
           "">,
  ];
  let constructor = "mlir::llh::createTransformLayoutPass()";
}

def LoadWeightPass : Pass<"load-weight","ModuleOp"> {
  let summary = "将WeightOp转为ConstOp";
  let description = [{
    1.用WeightOp代替Const是因为在xdsl上定义DensAttr时间太长。
    2.在IR图上加载DensAttr占用内存太大,不方便调试。
    3.可以转为runtime api加载Weight,减少编译模型文件的大小和编译时间。
  }];
}

//暂时不做
// def StoreConstToWeight : Pass<"store-const-to-weight","ModuleOp"> {
//   let summary = "将权重储存到文件中";
//   let constructor = "mlir::llh::createStoreConstToWeightPass()";
// }


def OperationlegalizationPass : Pass<"operation-legalization","ModuleOp"> {
  let summary = "合法化Op";
  let description = [{
    1.为布局敏感的Op添加布局属性;

    ```mlir
    %176 = "llh.conv"(%175, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
    "llh.symbolic_bind"(%174, %122, %123) <{expressions = #map5}> : (tensor<?x512x?x?xf32>, i64, i64) -> ()
    '''
    ===>
    ```mlir
    %176 = "llh.conv"(%175, %49) <{dilation = array<i64: 1, 1>, group = 1 : i64, kernel_shape = array<i64: 3, 3>, layout = #llh.Layout<NCHW>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<?x256x?x?xf32>, tensor<512x256x3x3xf32>) -> tensor<?x512x?x?xf32>
    '''
    

    2.将非法的Tensor与标量的计算合法化; 这是方便对接框架才出现的非法情况。

    ```mlir
    %9 = "llh.add"(%7, %8) : (tensor<?x?x224x224xf32>, i64) -> tensor<?x?x224x224xf32>
    '''
    ===>
    ```mlir
    %9 = "llh.add"(%9, %2) : (tensor<?x?x224x224xf32>, tensor<1xf32>) -> tensor<?x?x224x224xf32>
    '''
  }];
  let dependentDialects = [ "::mlir::shape::ShapeDialect",
                            "::mlir::math::MathDialect",
                            "::mlir::complex::ComplexDialect",
                            "::mlir::bufferization::BufferizationDialect",
                            "::mlir::cf::ControlFlowDialect"];
}

def RemoveRedundantOpsPass : Pass<"remove-redundant-ops","ModuleOp">{
  let summary = "去除冗余的Op";
  let description = [{
    1.去除从torch框架上传的符号表达op (符号会在其他Pass中生成,弃用该方案);
    2.将具有相似语义同意替换为1个,减少后续做优化的复杂度; flatten --> reshape , convbais --> conv + add
  }];
}

def ReshapeBeforeBraodcastPass : Pass<"reshape-before-braodcast","ModuleOp">{
  let summary = "在广播前插入reshape";
  let description = [{
    1.为了方便lowing到tosa,需要lowing前做融合。broadcast == reshape + tile;
  }];
}

def InsertBroadCastPass : Pass<"insert-broadcast","ModuleOp">{
  let summary = "插入braodcast";
  let description = [{
    
  }];
}

def MarkAotPass : Pass<"mark-aot","ModuleOp">{
  let summary = "标记aot算子";
  let description = [{
    
  }];
}

def DecomposeOpsPass : Pass<"decompose-ops","ModuleOp">{
  let summary = "分解算子";
  let description = [{
    
  }];
  let dependentDialects = [ "::mlir::llh::LLHDialect"];
}
#endif // LLH_PASS