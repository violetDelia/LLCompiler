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
#include "llcompiler/Dialect/Utility/Benefit.h"
namespace mlir::llh {
const size_t ReshapeBenefit = 100;
const size_t BroadcastBenefit = 99;
const size_t RefineOpBenefit = 98;
const size_t SinkOpBenfit = 97;
const size_t RemoveBenfit = 1;
}  // namespace mlir::llh