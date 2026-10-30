// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

#ifndef STRUCTURAL_OPERAND_OP_INTERFACE
#define STRUCTURAL_OPERAND_OP_INTERFACE

include "mlir/IR/OpBase.td"

def StructuralOperandOpInterface : OpInterface<"StructuralOperandOpInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Identifies operands whose values are inspected for compile-time semantics,
    including result shapes, axes, and control flow. This describes the operand
    role; it does not imply that ONNX requires the operand to be constant.
  }];

  let methods = [
    InterfaceMethod<"Return whether the operand has a structural role.",
        "bool", "isStructuralOperand", (ins "unsigned":$index)>
  ];
}

#endif // STRUCTURAL_OPERAND_OP_INTERFACE
