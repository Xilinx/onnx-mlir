// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#ifndef QUANTIZATION_BOUNDARY_OP_INTERFACE
#define QUANTIZATION_BOUNDARY_OP_INTERFACE

include "mlir/IR/OpBase.td"

def QuantizationBoundaryOpInterface : OpInterface<"QuantizationBoundaryOpInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Identifies operations that establish quantization or dequantization
    boundaries. This marker does not change purity or constant-foldability.
    Ordering-only rewrites can use it to preserve the position of a boundary
    represented inside a deferred recipe, just as for an explicit operation.

    External models allow other dialects to identify their boundaries without
    introducing a dependency from ONNX to those dialects.
  }];
}

#endif // QUANTIZATION_BOUNDARY_OP_INTERFACE
