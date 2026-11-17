// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#include "src/Interface/DeferredConstantOpInterface.hpp"

#include "src/Interface/DeferredConstantOpInterface.cpp.inc"

namespace mlir {

bool isDeferredConstant(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  auto deferred = dyn_cast<DeferredConstantOpInterface>(result.getOwner());
  return deferred &&
         deferred.isDeferredConstantResult(result.getResultNumber());
}

} // namespace mlir
