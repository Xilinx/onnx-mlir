// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=tosa.reshape
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --implicit-check-not=tosa.reshape
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=tosa.reshape

// Literal zero dimensions require allowzero, which the existing Reshape
// lowering does not support. Keep these reshapes in ONNX with correct shapes.
func.func @squeeze_empty(%input: tensor<1x0x4xf32>) -> tensor<0x4xf32> {
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %0 = "onnx.Squeeze"(%input, %axes) : (tensor<1x0x4xf32>, tensor<1xi64>) -> tensor<0x4xf32>
  return %0 : tensor<0x4xf32>
}

// CHECK-LABEL: func.func @squeeze_empty(
// CHECK-SAME: %[[INPUT:.*]]:
// CHECK: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[0, 4]> : tensor<2xi64>
// CHECK: %[[RESULT:.*]] = "onnx.Reshape"(%[[INPUT]], %[[SHAPE]]) {allowzero = 1 : si64} {{.*}} -> tensor<0x4xf32>
// CHECK: return %[[RESULT]] : tensor<0x4xf32>

// -----

func.func @squeeze_multiple_empty_dims(%input: tensor<1x1x0x0x4xf32>) -> tensor<0x0x4xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %0 = "onnx.Squeeze"(%input, %axes) : (tensor<1x1x0x0x4xf32>, tensor<2xi64>) -> tensor<0x0x4xf32>
  return %0 : tensor<0x0x4xf32>
}

// CHECK-LABEL: func.func @squeeze_multiple_empty_dims(
// CHECK-SAME: %[[INPUT:.*]]:
// CHECK: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[0, 0, 4]> : tensor<3xi64>
// CHECK: %[[RESULT:.*]] = "onnx.Reshape"(%[[INPUT]], %[[SHAPE]]) {allowzero = 1 : si64} {{.*}} -> tensor<0x0x4xf32>
// CHECK: return %[[RESULT]] : tensor<0x0x4xf32>

// -----

func.func @flatten_empty(%input: tensor<2x0x3xf32>) -> tensor<0x3xf32> {
  %0 = "onnx.Flatten"(%input) {axis = 2 : si64} : (tensor<2x0x3xf32>) -> tensor<0x3xf32>
  return %0 : tensor<0x3xf32>
}

// CHECK-LABEL: func.func @flatten_empty(
// CHECK-SAME: %[[INPUT:.*]]:
// CHECK: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[0, 3]> : tensor<2xi64>
// CHECK: %[[RESULT:.*]] = "onnx.Reshape"(%[[INPUT]], %[[SHAPE]]) {allowzero = 1 : si64} {{.*}} -> tensor<0x3xf32>
// CHECK: return %[[RESULT]] : tensor<0x3xf32>

// -----

func.func @mul_empty(%lhs: tensor<2x0xf32>, %rhs: tensor<0xf32>) -> tensor<2x0xf32> {
  %0 = "onnx.Mul"(%lhs, %rhs) : (tensor<2x0xf32>, tensor<0xf32>) -> tensor<2x0xf32>
  return %0 : tensor<2x0xf32>
}

// CHECK-LABEL: func.func @mul_empty(
// CHECK-SAME: %[[LHS:.*]]: tensor<2x0xf32>, %[[RHS:.*]]: tensor<0xf32>
// CHECK: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[1, 0]> : tensor<2xi64>
// CHECK: %[[RESHAPED:.*]] = "onnx.Reshape"(%[[RHS]], %[[SHAPE]]) {allowzero = 1 : si64} {{.*}} -> tensor<1x0xf32>
// CHECK: %[[RESULT:.*]] = tosa.mul %[[LHS]], %[[RESHAPED]]
// CHECK: return %[[RESULT]] : tensor<2x0xf32>
