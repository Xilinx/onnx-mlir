// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt %s --verify-diagnostics

func.func @unsupported_round_mode(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{has unsupported round_mode 'unknown'}}
  %result = "onnx.XFECast"(%input) {round_mode = "unknown", saturate = 0 : si64, to = bf16} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @invalid_saturate(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{requires 'saturate' to be 0 or 1}}
  %result = "onnx.XFECast"(%input) {round_mode = "toward_zero", saturate = 2 : si64, to = bf16} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @invalid_to(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{output element type does not match the 'to' attribute}}
  %result = "onnx.XFECast"(%input) {round_mode = "toward_zero", saturate = 0 : si64, to = f32} : (tensor<2xf32>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @onnx_cast_type_domain(%input: tensor<2xf16>) -> tensor<2xf32> {
  %result = "onnx.XFECast"(%input) {round_mode = "nearest_even", saturate = 0 : si64, to = f32} : (tensor<2xf16>) -> tensor<2xf32>
  return %result : tensor<2xf32>
}
