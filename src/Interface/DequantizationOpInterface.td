// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#ifndef DEQUANTIZATION_OP_INTERFACE
#define DEQUANTIZATION_OP_INTERFACE

include "mlir/IR/OpBase.td"

def DequantizationOpInterface : OpInterface<"DequantizationOpInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    The operation returns dequantized values, including when it performs both
    quantization and dequantization.
  }];
}

#endif // DEQUANTIZATION_OP_INTERFACE
