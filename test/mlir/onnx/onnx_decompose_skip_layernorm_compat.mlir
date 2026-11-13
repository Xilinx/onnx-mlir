// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx=enable-matmulnbits-decompose %s -split-input-file | FileCheck %s

func.func @skip_layernorm_compat_none_bias(%arg0: tensor<1x2x4xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<4xf32>) -> tensor<1x2x4xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Custom"(%arg0, %arg1, %arg2, %0, %0) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization"} : (tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<4xf32>, none, none) -> tensor<1x2x4xf32>
  return %1 : tensor<1x2x4xf32>
}
// CHECK-LABEL: func.func @skip_layernorm_compat_none_bias(
// CHECK: "onnx.Add"
// CHECK-NOT: "onnx.Add"
// CHECK: "onnx.LayerNormalization"
// CHECK: return

// -----
