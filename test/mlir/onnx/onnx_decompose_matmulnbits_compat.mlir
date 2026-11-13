// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx=enable-matmulnbits-decompose %s -split-input-file | FileCheck %s

func.func @default_midpoint(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.Custom"(%arg0, %0, %1) {K = 16 : si64, N = 2 : si64, accuracy_level = 4 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits", weight_prepacked = 0 : si64} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>) -> tensor<1x2x2xf32>
  return %2 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @default_midpoint(
// CHECK: onnx.Constant dense<8> : tensor<1x2x1xui8>
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.MatMul"
// CHECK: return

// -----

func.func @none_midpoint(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.Custom"(%arg0, %0, %1, %2) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>, none) -> tensor<1x2x2xf32>
  return %3 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @none_midpoint(
// CHECK: onnx.Constant dense<8> : tensor<1x2x1xui8>
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.MatMul"
// CHECK: return

// -----

func.func @midpoint_bias(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = onnx.Constant dense<3.000000e+00> : tensor<2xf32>
  %4 = "onnx.Custom"(%arg0, %0, %1, %2, %2, %3) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>, none, none, tensor<2xf32>) -> tensor<1x2x2xf32>
  return %4 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @midpoint_bias(
// CHECK: "onnx.MatMul"
// CHECK: "onnx.Add"
// CHECK: return

// -----

func.func @explicit_midpoint(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = onnx.Constant dense<136> : tensor<2x1xui8>
  %4 = "onnx.Custom"(%arg0, %0, %1, %3) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>, tensor<2x1xui8>) -> tensor<1x2x2xf32>
  return %4 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @explicit_midpoint(
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.MatMul"
// CHECK: return

// -----

func.func @prepacked_sm80(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.Custom"(%arg0, %0, %1) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits", weight_prepacked = 1 : si64} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>) -> tensor<1x2x2xf32>
  return %2 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @prepacked_sm80(
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return

// -----

func.func @nonconstant_B(%arg0: tensor<1x2x16xf32>, %arg1: tensor<2x1x8xui8>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %1 = "onnx.Custom"(%arg0, %arg1, %0) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>) -> tensor<1x2x2xf32>
  return %1 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @nonconstant_B(
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return

// -----

func.func @empty_B(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x0x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.Custom"(%arg0, %0, %1) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x0x8xui8>, tensor<2x1xf32>) -> tensor<1x2x2xf32>
  return %2 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @empty_B(
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return

// -----

func.func @float_zp(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = onnx.Constant dense<8.000000e+00> : tensor<2x1xf32>
  %4 = "onnx.Custom"(%arg0, %0, %1, %3) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<1x2x2xf32>
  return %4 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @float_zp(
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return

// -----

func.func @dynamic_zp(%arg0: tensor<1x2x16xf32>, %arg1: tensor<2x1xui8>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x8xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.Custom"(%arg0, %0, %1, %arg1) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x8xui8>, tensor<2x1xf32>, tensor<2x1xui8>) -> tensor<1x2x2xf32>
  return %3 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @dynamic_zp(
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return

// -----

func.func @bad_B_shape(%arg0: tensor<1x2x16xf32>) -> tensor<1x2x2xf32> {
  %0 = onnx.Constant dense<136> : tensor<2x1x7xui8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<2x1xf32>
  %2 = "onnx.Custom"(%arg0, %0, %1) {K = 16 : si64, N = 2 : si64, block_size = 16 : si64, domain_name = "com.microsoft", function_name = "MatMulNBits"} : (tensor<1x2x16xf32>, tensor<2x1x7xui8>, tensor<2x1xf32>) -> tensor<1x2x2xf32>
  return %2 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @bad_B_shape(
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return

// -----
