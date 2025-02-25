//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {
class DialectRegistry;

namespace llh{
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace cf
} // namespace mlir

#endif // INCLUDE_LLCOMPILER_DIALECT_LLH_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
