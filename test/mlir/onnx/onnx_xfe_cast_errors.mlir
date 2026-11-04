// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt %s --verify-diagnostics

func.func @unsupported_policy(%input: tensor<2xf32>) -> tensor<2xbf16> {
  // expected-error @+1 {{has unsupported policy 'unknown'}}
  %result = onnx.XFECast %input policy = "unknown" : tensor<2xf32> to tensor<2xbf16>
  return %result : tensor<2xbf16>
}
