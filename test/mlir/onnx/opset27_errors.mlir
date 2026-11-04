// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.
// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

func.func @bitcast_width(%x: tensor<2xf32>) -> tensor<2xi64> {
  // expected-error@+1 {{input and output element types must have equal bit width}}
  %y = "onnx.BitCast"(%x) {to = i64} : (tensor<2xf32>) -> tensor<2xi64>
  return %y : tensor<2xi64>
}

// -----

func.func @cumprod_axis(%x: tensor<2xf32>, %axis: tensor<1xi64>) -> tensor<2xf32> {
  // expected-error@+1 {{'axis' must be a scalar tensor}}
  %y = "onnx.CumProd"(%x, %axis) : (tensor<2xf32>, tensor<1xi64>) -> tensor<2xf32>
  return %y : tensor<2xf32>
}

// -----

func.func @tensor_scatter_mode(%cache: tensor<2x4xf32>, %x: tensor<2x1xf32>,
    %indices: tensor<2xi64>) -> tensor<2x4xf32> {
  // expected-error@+1 {{'mode' must be 'linear' or 'circular'}}
  %y = "onnx.TensorScatter"(%cache, %x, %indices) {mode = "invalid"}
      : (tensor<2x4xf32>, tensor<2x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  return %y : tensor<2x4xf32>
}
