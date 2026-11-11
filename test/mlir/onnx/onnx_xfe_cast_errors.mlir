// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt %s --verify-diagnostics

func.func @unsupported_round_mode(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{has unsupported round_mode 'unknown'}}
  %result = "onnx.XFECast"(%input) {nan_mode = "propagate", round_mode = "unknown", saturation_mode = "none", to = bf16} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @invalid_saturation_mode(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{has unsupported saturation_mode 'unknown'}}
  %result = "onnx.XFECast"(%input) {nan_mode = "propagate", round_mode = "toward_zero", saturation_mode = "unknown", to = bf16} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @invalid_nan_mode(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{has unsupported nan_mode 'unknown'}}
  %result = "onnx.XFECast"(%input) {nan_mode = "unknown", round_mode = "toward_zero", saturation_mode = "none", to = bf16} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @invalid_to(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{output element type does not match the 'to' attribute}}
  %result = "onnx.XFECast"(%input) {nan_mode = "propagate", round_mode = "toward_zero", saturation_mode = "none", to = f32} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @onnx_cast_type_domain(%input: tensor<2xf16>) -> tensor<2xf32> {
  %result = "onnx.XFECast"(%input) {nan_mode = "propagate", round_mode = "nearest_even", saturation_mode = "none", to = f32} : (tensor<2xf16>) -> tensor<2xf32>
  return %result : tensor<2xf32>
}
