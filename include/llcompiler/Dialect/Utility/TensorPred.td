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
#ifndef TENSORPRED
#define TENSORPRED
include "mlir/IR/Constraints.td"
class Tensor_Rank_Is<int n>: 
    Constraint< And<[   CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getRank() == " # n>]
                    >,"tensor rank is " # n>;
def Is_2D_Tensor :Tensor_Rank_Is<2>;
def Is_3D_Tensor :Tensor_Rank_Is<3>;


#endif // TENSORPRED