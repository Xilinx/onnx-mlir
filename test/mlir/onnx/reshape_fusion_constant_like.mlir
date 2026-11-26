// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize --cse %s -split-input-file | FileCheck %s

// Mixed constant producers must not prevent inverse reshapes from folding.
func.func @mixed_constant_inverse(%input: tensor<1x48x8x8xf32>) -> tensor<1x48x8x8xf32> {
  %shape1 = onnx.Constant dense<[48, 8, 8]> : tensor<3xi64>
  %shape2 = "tosa.const"() <{value = dense<[1, 48, 8, 8]> : tensor<4xi64>}> : () -> tensor<4xi64>
  %first = "onnx.Reshape"(%input, %shape1) {allowzero = 0 : si64} : (tensor<1x48x8x8xf32>, tensor<3xi64>) -> tensor<48x8x8xf32>
  %second = "onnx.Reshape"(%first, %shape2) {allowzero = 0 : si64} : (tensor<48x8x8xf32>, tensor<4xi64>) -> tensor<1x48x8x8xf32>
  return %second : tensor<1x48x8x8xf32>
}

// CHECK-LABEL: func.func @mixed_constant_inverse
// CHECK-SAME: %[[INPUT:.*]]: tensor<1x48x8x8xf32>
// CHECK-NOT: onnx.Reshape
// CHECK: return %[[INPUT]]

// -----

// Fusion must retain the final shape when the combined reshape is not identity.
func.func @arith_constant_fusion(%input: tensor<2x3x4xf32>) -> tensor<4x6xf32> {
  %shape1 = arith.constant dense<[6, 4]> : tensor<2xi64>
  %shape2 = arith.constant dense<[4, 6]> : tensor<2xi64>
  %first = "onnx.Reshape"(%input, %shape1) {allowzero = 0 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<6x4xf32>
  %second = "onnx.Reshape"(%first, %shape2) {allowzero = 0 : si64} : (tensor<6x4xf32>, tensor<2xi64>) -> tensor<4x6xf32>
  return %second : tensor<4x6xf32>
}

// CHECK-LABEL: func.func @arith_constant_fusion
// CHECK-SAME: %[[INPUT:.*]]: tensor<2x3x4xf32>
// CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[4, 6]> : tensor<2xi64>
// CHECK: %[[RESULT:.*]] = "onnx.Reshape"(%[[INPUT]], %[[SHAPE]])
// CHECK-NOT: onnx.Reshape
// CHECK: return %[[RESULT]]

