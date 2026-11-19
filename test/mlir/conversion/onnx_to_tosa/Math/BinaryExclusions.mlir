// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Add,Sub,Mul,Div,Pow,Min,Max,Equal,Greater,GreaterOrEqual,Less,LessOrEqual,And,Or,Xor,BitwiseAnd,BitwiseOr,BitwiseXor" -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,EXCLUDED --implicit-check-not=tosa.
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Add,Sub,Mul,Div,Pow,Min,Max,Equal,Greater,GreaterOrEqual,Less,LessOrEqual,And,Or,Xor,BitwiseAnd,BitwiseOr,BitwiseXor" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not=onnx.

func.func @add(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Add"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @add(
// EXCLUDED: %[[R:.*]] = "onnx.Add"
// LOWERED: %[[R:.*]] = tosa.add
// CHECK: return %[[R]] : tensor<2x3xf32>

// -----

func.func @sub(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Sub"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @sub(
// EXCLUDED: %[[R:.*]] = "onnx.Sub"
// LOWERED: %[[R:.*]] = tosa.sub
// CHECK: return %[[R]] : tensor<2x3xf32>

// -----

func.func @mul(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Mul"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul(
// EXCLUDED: %[[R:.*]] = "onnx.Mul"
// LOWERED: %[[R:.*]] = tosa.mul
// CHECK: return %[[R]] : tensor<2x3xf32>

// -----

func.func @div(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %r = "onnx.Div"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @div(
// EXCLUDED: %[[R:.*]] = "onnx.Div"
// LOWERED: %[[R:.*]] = tosa.int_div
// CHECK: return %[[R]] : tensor<2x3xi32>

// -----

func.func @pow(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Pow"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @pow(
// EXCLUDED: %[[R:.*]] = "onnx.Pow"
// LOWERED: %[[R:.*]] = tosa.pow
// CHECK: return %[[R]] : tensor<2x3xf32>

// -----

func.func @min(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Min"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min(
// EXCLUDED: %[[R:.*]] = "onnx.Min"
// LOWERED: %[[R:.*]] = tosa.minimum
// CHECK: return %[[R]] : tensor<2x3xf32>

// -----

func.func @max(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Max"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max(
// EXCLUDED: %[[R:.*]] = "onnx.Max"
// LOWERED: %[[R:.*]] = tosa.maximum
// CHECK: return %[[R]] : tensor<2x3xf32>

// -----

func.func @equal(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %r = "onnx.Equal"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @equal(
// EXCLUDED: %[[R:.*]] = "onnx.Equal"
// LOWERED: %[[R:.*]] = tosa.equal
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @greater(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %r = "onnx.Greater"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @greater(
// EXCLUDED: %[[R:.*]] = "onnx.Greater"
// LOWERED: %[[R:.*]] = tosa.greater
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @greaterorequal(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %r = "onnx.GreaterOrEqual"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @greaterorequal(
// EXCLUDED: %[[R:.*]] = "onnx.GreaterOrEqual"
// LOWERED: %[[R:.*]] = tosa.greater_equal
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @less(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %r = "onnx.Less"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @less(
// EXCLUDED: %[[R:.*]] = "onnx.Less"
// LOWERED: %[[R:.*]] = tosa.greater
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @lessorequal(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xi1> {
  %r = "onnx.LessOrEqual"(%a, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @lessorequal(
// EXCLUDED: %[[R:.*]] = "onnx.LessOrEqual"
// LOWERED: %[[R:.*]] = tosa.greater_equal
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @and(%a: tensor<2x3xi1>, %b: tensor<2x3xi1>) -> tensor<2x3xi1> {
  %r = "onnx.And"(%a, %b) : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @and(
// EXCLUDED: %[[R:.*]] = "onnx.And"
// LOWERED: %[[R:.*]] = tosa.logical_and
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @or(%a: tensor<2x3xi1>, %b: tensor<2x3xi1>) -> tensor<2x3xi1> {
  %r = "onnx.Or"(%a, %b) : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @or(
// EXCLUDED: %[[R:.*]] = "onnx.Or"
// LOWERED: %[[R:.*]] = tosa.logical_or
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @xor(%a: tensor<2x3xi1>, %b: tensor<2x3xi1>) -> tensor<2x3xi1> {
  %r = "onnx.Xor"(%a, %b) : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @xor(
// EXCLUDED: %[[R:.*]] = "onnx.Xor"
// LOWERED: %[[R:.*]] = tosa.logical_xor
// CHECK: return %[[R]] : tensor<2x3xi1>

// -----

func.func @bitwiseand(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %r = "onnx.BitwiseAnd"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @bitwiseand(
// EXCLUDED: %[[R:.*]] = "onnx.BitwiseAnd"
// LOWERED: %[[R:.*]] = tosa.bitwise_and
// CHECK: return %[[R]] : tensor<2x3xi32>

// -----

func.func @bitwiseor(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %r = "onnx.BitwiseOr"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @bitwiseor(
// EXCLUDED: %[[R:.*]] = "onnx.BitwiseOr"
// LOWERED: %[[R:.*]] = tosa.bitwise_or
// CHECK: return %[[R]] : tensor<2x3xi32>

// -----

func.func @bitwisexor(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %r = "onnx.BitwiseXor"(%a, %b) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @bitwisexor(
// EXCLUDED: %[[R:.*]] = "onnx.BitwiseXor"
// LOWERED: %[[R:.*]] = tosa.bitwise_xor
// CHECK: return %[[R]] : tensor<2x3xi32>
