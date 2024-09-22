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

def TransformLayoutToNHWC : Pass<"transform-layout-to-nhwc","ModuleOp"> {
  let summary = "trans some llh op layout to nhwc for lowing to tosa";
  let description = [{
  }];
  let constructor = "mlir::llh::createTransformLayoutToNHWCPass()";
  let dependentDialects = [ "mlir::ex::IRExtensionDialect"];
}

def LoadWeight : Pass<"load-weight","ModuleOp"> {
  let summary = "load weight to const";
  let description = [{
  }];
  let constructor = "mlir::llh::createLoadWeightPass()";
}

def StoreConstToWeight : Pass<"store-const-to-weight","ModuleOp"> {
  let summary = "store constant to weight";
  let constructor = "mlir::llh::createStoreConstToWeightPass()";
}

def GenerateSymbol : Pass<"generate-symbol","ModuleOp"> {
  let summary = "initialization of the analysis and remove shape bind";
  let constructor = "mlir::llh::createGenerateSymbolPass()";

}

#endif // LLH_PASS