// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,EXCLUDED --implicit-check-not=tosa.reshape
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not='"onnx.Reshape"'

// Flatten and Squeeze emit ONNX reshapes, which lower transitively when
// Reshape conversion is enabled.
func.func @flatten(%input: tensor<2x3x4xf32>) -> tensor<2x12xf32> {
  %0 = "onnx.Flatten"(%input) {axis = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<2x12xf32>
  return %0 : tensor<2x12xf32>
}

// CHECK-LABEL: func.func @flatten(
// CHECK-SAME: %[[INPUT:.*]]:
// LOWERED: %[[FLAT:.*]] = tosa.reshape %[[INPUT]] {new_shape = array<i64: 2, 12>}
// EXCLUDED: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[2, 12]> : tensor<2xi64>
// EXCLUDED: %[[FLAT:.*]] = "onnx.Reshape"(%[[INPUT]], %[[SHAPE]]) {{.*}} -> tensor<2x12xf32>
// CHECK: return %[[FLAT]] : tensor<2x12xf32>

// -----

func.func @squeeze(%input: tensor<1x3x1x4xf32>) -> tensor<3x4xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0 = "onnx.Squeeze"(%input, %axes) : (tensor<1x3x1x4xf32>, tensor<2xi64>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @squeeze(
// CHECK-SAME: %[[INPUT:.*]]:
// LOWERED: %[[SQUEEZED:.*]] = tosa.reshape %[[INPUT]] {new_shape = array<i64: 3, 4>}
// EXCLUDED: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[3, 4]> : tensor<2xi64>
// EXCLUDED: %[[SQUEEZED:.*]] = "onnx.Reshape"(%[[INPUT]], %[[SHAPE]]) {{.*}} -> tensor<3x4xf32>
// CHECK: return %[[SQUEEZED]] : tensor<3x4xf32>

// -----

func.func @reshape(%input: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %shape = "onnx.Constant"() {value = dense<[6, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>
}

// CHECK-LABEL: func.func @reshape(
// CHECK-SAME: %[[INPUT:.*]]:
// LOWERED: %[[RESHAPED:.*]] = tosa.reshape %[[INPUT]] {new_shape = array<i64: 6, 4>}
// EXCLUDED: %[[RESHAPED:.*]] = "onnx.Reshape"(%[[INPUT]], %{{.*}}) {{.*}} -> tensor<6x4xf32>
// CHECK: return %[[RESHAPED]] : tensor<6x4xf32>
