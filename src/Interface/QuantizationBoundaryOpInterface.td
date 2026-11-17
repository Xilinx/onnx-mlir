// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#ifndef QUANTIZATION_BOUNDARY_OP_INTERFACE
#define QUANTIZATION_BOUNDARY_OP_INTERFACE

include "mlir/IR/OpBase.td"

def QuantizationBoundaryOpInterface : OpInterface<"QuantizationBoundaryOpInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Identifies operations that quantize, dequantize, or perform both conversions.
  }];
}

#endif // QUANTIZATION_BOUNDARY_OP_INTERFACE
