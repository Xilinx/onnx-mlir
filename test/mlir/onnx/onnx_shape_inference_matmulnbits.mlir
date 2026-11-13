// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// MatMulNBits (com.microsoft) is imported as a CustomOp, so its result type
// comes from the model annotation. The result shape is the shape of A with the
// last dimension (K) replaced by N, which resolves a dimension the model left
// symbolic.
func.func @unknown_result_dim(%a: tensor<1x1x32xf32>, %b: tensor<4x2x8xui8>, %scales: tensor<4x2xf32>) -> tensor<1x?x4xf32> {
  %out = "onnx.Custom"(%a, %b, %scales) {
    domain_name = "com.microsoft",
    function_name = "MatMulNBits",
    K = 32 : si64,
    N = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  } : (tensor<1x1x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>) -> tensor<1x?x4xf32>
  return %out : tensor<1x?x4xf32>
}

// CHECK-LABEL: func.func @unknown_result_dim
// CHECK: "onnx.Custom"
// CHECK-SAME: -> tensor<1x1x4xf32>

// -----

// A dimension that is unknown in A stays unknown; the others are still taken
// from A, and the last one from N.
func.func @dynamic_input_dim(%a: tensor<?x3x32xf32>, %b: tensor<4x2x8xui8>, %scales: tensor<4x2xf32>) -> tensor<*xf32> {
  %out = "onnx.Custom"(%a, %b, %scales) {
    domain_name = "com.microsoft",
    function_name = "MatMulNBits",
    K = 32 : si64,
    N = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  } : (tensor<?x3x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>) -> tensor<*xf32>
  return %out : tensor<*xf32>
}

// CHECK-LABEL: func.func @dynamic_input_dim
// CHECK: "onnx.Custom"
// CHECK-SAME: -> tensor<?x3x4xf32>

// -----

// Only A is read: an omitted optional input does not block the inference.
func.func @none_optional_input(%a: tensor<2x32xf32>, %b: tensor<4x2x8xui8>, %scales: tensor<4x2xf32>, %bias: tensor<4xf32>) -> tensor<?x?xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %out = "onnx.Custom"(%a, %b, %scales, %none, %none, %bias) {
    domain_name = "com.microsoft",
    function_name = "MatMulNBits",
    K = 32 : si64,
    N = 4 : si64,
    bits = 4 : si64,
    block_size = 16 : si64
  } : (tensor<2x32xf32>, tensor<4x2x8xui8>, tensor<4x2xf32>, none, none, tensor<4xf32>) -> tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @none_optional_input
// CHECK: "onnx.Custom"
// CHECK-SAME: -> tensor<2x4xf32>

// -----

// Another CustomOp without a shape_infer_pattern keeps its annotated type.
func.func @other_custom_op(%a: tensor<1x1x32xf32>) -> tensor<1x?x4xf32> {
  %out = "onnx.Custom"(%a) {
    domain_name = "com.microsoft",
    function_name = "SomeOtherOp",
    N = 4 : si64
  } : (tensor<1x1x32xf32>) -> tensor<1x?x4xf32>
  return %out : tensor<1x?x4xf32>
}

// CHECK-LABEL: func.func @other_custom_op
// CHECK: "onnx.Custom"
// CHECK-SAME: -> tensor<1x?x4xf32>
