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

#ifndef INCLUDE_LLCOMPILER_TRANSFORMLIBRARY_LIBRARYENTRY_H_
#define INCLUDE_LLCOMPILER_TRANSFORMLIBRARY_LIBRARYENTRY_H_

#define __LLC_TRANSFORM_LINALG_GENERALIZE__ "linalg_generalize"
#define __LLC_TRANSFORM_LINALG_SPECIALIZE__ "linalg_specialize"
#define __LLC_TRANSFORM_LINALG_FLATTEN__ "linalg_flatten_elementwise"
#define __LLC_TRANSFORM_LINALG_BASIC_FUSE__ "linalg_basic_fuse"
#define __LLC_TRANSFORM_LINALG_BASIC_VECTORIZATION__ \
  "linalg_basic_vectorization"
#define __LLC_TRANSFORM_LINALG_BASIC_BUFFERIZATION__ "linalg_bufferization"
#define __LLC_TRANSFORM_LINALG_BASIC_ANALYSIS__ "linalg_analysis"
#define __LLC_TRANSFORM_MHLO_BASIC_OPT__ "mhlo_basic_opt"
#define __LLC_TRANSFORM_MHLO_TO_LINALG__ "mhlo_to_linalg"
#define __LLC_TRANSFORM_MHLO_BUFFERIZE__ "mhlo_one_shot_bufferize"
#define __LLC_TRANSFORM_HLO_TO_LINALG__ "stablehlo_to_linalg"
#define __LLC_TRANSFORM_HLO_BASIC_OPT__ "stablehlo_basic_opt"
#define __LLC_TRANSFORM_TENSOR_BASIC_OPT__ "tensor_basic_opt"
#define __LLC_TRANSFORM_LLVM_LOWING__ "lowing_to_llvm"
#define __LLC_TRANSFORM_LLVM_BASIC_OPT__ "llvm_basic_opt"
#define __LLC_TRANSFORM_MEMREF_BASIC_OPT__ "memref_basic_opt"

#endif  // INCLUDE_LLCOMPILER_TRANSFORMLIBRARY_LIBRARYENTRY_H_
