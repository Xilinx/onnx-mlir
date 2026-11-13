// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx=enable-matmulnbits-decompose %s -split-input-file | FileCheck %s

func.func @rope_unit_scale(%arg0: tensor<1x2x2x4xf32>, %arg1: tensor<1x2xi64>, %arg2: tensor<16x2xf32>, %arg3: tensor<16x2xf32>) -> tensor<1x2x2x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3) {domain_name = "com.microsoft", function_name = "RotaryEmbedding", scale = 1.000000e+00 : f32} : (tensor<1x2x2x4xf32>, tensor<1x2xi64>, tensor<16x2xf32>, tensor<16x2xf32>) -> tensor<1x2x2x4xf32>
  return %0 : tensor<1x2x2x4xf32>
}
// CHECK-LABEL: func.func @rope_unit_scale(
// CHECK: "onnx.RotaryEmbedding"
// CHECK: return

// -----

func.func @rope_nonunit(%arg0: tensor<1x2x2x4xf32>, %arg1: tensor<1x2xi64>, %arg2: tensor<16x2xf32>, %arg3: tensor<16x2xf32>) -> tensor<1x2x2x4xf32> {
  %0 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3) {domain_name = "com.microsoft", function_name = "RotaryEmbedding", scale = 2.000000e+00 : f32} : (tensor<1x2x2x4xf32>, tensor<1x2xi64>, tensor<16x2xf32>, tensor<16x2xf32>) -> tensor<1x2x2x4xf32>
  return %0 : tensor<1x2x2x4xf32>
}
// CHECK-LABEL: func.func @rope_nonunit(
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.RotaryEmbedding"
// CHECK: return

// -----
