// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt %s -o /dev/null

// Internal type extensions admit narrow integer counts. Admission does not
// imply that every target has a kernel for every type or overflow behavior.

func.func @reduce_sum_i8(%input: tensor<2x12xi8>) -> (tensor<2xi8>, tensor<2xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %sum = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceSumV11"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xi8>) -> tensor<2xi8>
  return %sum, %legacy : tensor<2xi8>, tensor<2xi8>
}

func.func @reduce_sum_ui8(%input: tensor<2x12xui8>) -> (tensor<2xui8>, tensor<2xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %sum = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceSumV11"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xui8>) -> tensor<2xui8>
  return %sum, %legacy : tensor<2xui8>, tensor<2xui8>
}

func.func @reduce_sum_i16(%input: tensor<2x12xi16>) -> (tensor<2xi16>, tensor<2xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %sum = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceSumV11"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xi16>) -> tensor<2xi16>
  return %sum, %legacy : tensor<2xi16>, tensor<2xi16>
}

func.func @reduce_sum_ui16(%input: tensor<2x12xui16>) -> (tensor<2xui16>, tensor<2xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %sum = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceSumV11"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xui16>) -> tensor<2xui16>
  return %sum, %legacy : tensor<2xui16>, tensor<2xui16>
}

func.func @reduce_sum_i32(%input: tensor<2x12xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %sum = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceSumV11"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xi32>) -> tensor<2xi32>
  return %sum, %legacy : tensor<2xi32>, tensor<2xi32>
}

func.func @reduce_sum_ui32(%input: tensor<2x12xui32>) -> (tensor<2xui32>, tensor<2xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %sum = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceSumV11"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xui32>) -> tensor<2xui32>
  return %sum, %legacy : tensor<2xui32>, tensor<2xui32>
}
