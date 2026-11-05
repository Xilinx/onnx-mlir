// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#ifndef ONNX_MLIR_DEFERRED_CONSTANT_OP_INTERFACE_H
#define ONNX_MLIR_DEFERRED_CONSTANT_OP_INTERFACE_H

#include "mlir/IR/OpDefinition.h"

#include "src/Interface/DeferredConstantOpInterface.hpp.inc"

namespace mlir {

/// Return whether `value` is an operation result with deferred constant
/// semantics. This query does not read or materialize the value.
bool isDeferredConstant(Value value);

} // namespace mlir

#endif // ONNX_MLIR_DEFERRED_CONSTANT_OP_INTERFACE_H
