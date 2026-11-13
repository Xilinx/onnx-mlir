// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx=enable-matmulnbits-decompose %s -split-input-file | FileCheck %s

func.func @fusedconv_identity(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_identity(
// CHECK: "onnx.Conv"
// CHECK-NOT: "onnx.Relu"
// CHECK: return

// -----

func.func @fusedconv_none_z(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Custom"(%arg0, %arg1, %0, %0) {domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>, none, none) -> tensor<1x1x4x4xf32>
  return %1 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_none_z(
// CHECK: "onnx.Conv"
// CHECK-NOT: "onnx.Relu"
// CHECK: return

// -----

func.func @fusedconv_relu(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {activation = "Relu", domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_relu(
// CHECK: "onnx.Conv"
// CHECK: "onnx.Relu"
// CHECK: return

// -----

func.func @fusedconv_missing_leaky(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {activation = "LeakyRelu", domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_missing_leaky(
// CHECK-NOT: "onnx.Conv"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.Conv"
// CHECK: return

// -----

func.func @fusedconv_short_clip(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {activation = "Clip", activation_params = [0.000000e+00 : f32], domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_short_clip(
// CHECK-NOT: "onnx.Conv"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.Conv"
// CHECK: return

// -----

func.func @fusedconv_bad_activation(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {activation = 1 : si64, domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_bad_activation(
// CHECK-NOT: "onnx.Conv"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.Conv"
// CHECK: return

// -----

func.func @fusedconv_integer_params(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1) {activation = "Relu", activation_params = [0 : si64], domain_name = "com.microsoft", function_name = "FusedConv"} : (tensor<1x1x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @fusedconv_integer_params(
// CHECK-NOT: "onnx.Conv"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.Conv"
// CHECK: return

// -----
