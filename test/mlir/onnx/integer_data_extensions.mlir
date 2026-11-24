// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// RUN: onnx-mlir-opt %s -o /dev/null

// Verify numeric data admission for smaller integer types.

func.func @cumsum_i8(%input: tensor<2x3xi8>) -> tensor<2x3xi8> {
  %axis = onnx.Constant dense<1> : tensor<i64>
  %result = "onnx.CumSum"(%input, %axis) : (tensor<2x3xi8>, tensor<i64>) -> tensor<2x3xi8>
  return %result : tensor<2x3xi8>
}

func.func @cumsum_ui8(%input: tensor<2x3xui8>) -> tensor<2x3xui8> {
  %axis = onnx.Constant dense<1> : tensor<i64>
  %result = "onnx.CumSum"(%input, %axis) : (tensor<2x3xui8>, tensor<i64>) -> tensor<2x3xui8>
  return %result : tensor<2x3xui8>
}

func.func @cumsum_i16(%input: tensor<2x3xi16>) -> tensor<2x3xi16> {
  %axis = onnx.Constant dense<1> : tensor<i64>
  %result = "onnx.CumSum"(%input, %axis) : (tensor<2x3xi16>, tensor<i64>) -> tensor<2x3xi16>
  return %result : tensor<2x3xi16>
}

func.func @cumsum_ui16(%input: tensor<2x3xui16>) -> tensor<2x3xui16> {
  %axis = onnx.Constant dense<1> : tensor<i64>
  %result = "onnx.CumSum"(%input, %axis) : (tensor<2x3xui16>, tensor<i64>) -> tensor<2x3xui16>
  return %result : tensor<2x3xui16>
}

func.func @cumsum_i32(%input: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %axis = onnx.Constant dense<1> : tensor<i64>
  %result = "onnx.CumSum"(%input, %axis) : (tensor<2x3xi32>, tensor<i64>) -> tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

func.func @cumsum_ui32(%input: tensor<2x3xui32>) -> tensor<2x3xui32> {
  %axis = onnx.Constant dense<1> : tensor<i64>
  %result = "onnx.CumSum"(%input, %axis) : (tensor<2x3xui32>, tensor<i64>) -> tensor<2x3xui32>
  return %result : tensor<2x3xui32>
}

func.func @prelu_i8(%input: tensor<2x3xi8>, %slope: tensor<3xi8>) -> tensor<2x3xi8> {
  %result = "onnx.PRelu"(%input, %slope) : (tensor<2x3xi8>, tensor<3xi8>) -> tensor<2x3xi8>
  return %result : tensor<2x3xi8>
}

func.func @prelu_ui8(%input: tensor<2x3xui8>, %slope: tensor<3xui8>) -> tensor<2x3xui8> {
  %result = "onnx.PRelu"(%input, %slope) : (tensor<2x3xui8>, tensor<3xui8>) -> tensor<2x3xui8>
  return %result : tensor<2x3xui8>
}

func.func @prelu_i16(%input: tensor<2x3xi16>, %slope: tensor<3xi16>) -> tensor<2x3xi16> {
  %result = "onnx.PRelu"(%input, %slope) : (tensor<2x3xi16>, tensor<3xi16>) -> tensor<2x3xi16>
  return %result : tensor<2x3xi16>
}

func.func @prelu_ui16(%input: tensor<2x3xui16>, %slope: tensor<3xui16>) -> tensor<2x3xui16> {
  %result = "onnx.PRelu"(%input, %slope) : (tensor<2x3xui16>, tensor<3xui16>) -> tensor<2x3xui16>
  return %result : tensor<2x3xui16>
}

func.func @prelu_i32(%input: tensor<2x3xi32>, %slope: tensor<3xi32>) -> tensor<2x3xi32> {
  %result = "onnx.PRelu"(%input, %slope) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

func.func @prelu_ui32(%input: tensor<2x3xui32>, %slope: tensor<3xui32>) -> tensor<2x3xui32> {
  %result = "onnx.PRelu"(%input, %slope) : (tensor<2x3xui32>, tensor<3xui32>) -> tensor<2x3xui32>
  return %result : tensor<2x3xui32>
}

func.func @pow_i8(%input: tensor<2x3xi8>, %exponent: tensor<f32>) -> tensor<2x3xi8> {
  %result = "onnx.Pow"(%input, %exponent) : (tensor<2x3xi8>, tensor<f32>) -> tensor<2x3xi8>
  return %result : tensor<2x3xi8>
}

func.func @pow_ui8(%input: tensor<2x3xui8>, %exponent: tensor<f32>) -> tensor<2x3xui8> {
  %result = "onnx.Pow"(%input, %exponent) : (tensor<2x3xui8>, tensor<f32>) -> tensor<2x3xui8>
  return %result : tensor<2x3xui8>
}

func.func @pow_i16(%input: tensor<2x3xi16>, %exponent: tensor<f32>) -> tensor<2x3xi16> {
  %result = "onnx.Pow"(%input, %exponent) : (tensor<2x3xi16>, tensor<f32>) -> tensor<2x3xi16>
  return %result : tensor<2x3xi16>
}

func.func @pow_ui16(%input: tensor<2x3xui16>, %exponent: tensor<f32>) -> tensor<2x3xui16> {
  %result = "onnx.Pow"(%input, %exponent) : (tensor<2x3xui16>, tensor<f32>) -> tensor<2x3xui16>
  return %result : tensor<2x3xui16>
}

func.func @pow_i32(%input: tensor<2x3xi32>, %exponent: tensor<f32>) -> tensor<2x3xi32> {
  %result = "onnx.Pow"(%input, %exponent) : (tensor<2x3xi32>, tensor<f32>) -> tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

func.func @pow_ui32(%input: tensor<2x3xui32>, %exponent: tensor<f32>) -> tensor<2x3xui32> {
  %result = "onnx.Pow"(%input, %exponent) : (tensor<2x3xui32>, tensor<f32>) -> tensor<2x3xui32>
  return %result : tensor<2x3xui32>
}

func.func @range_i8() -> tensor<3xi8> {
  %start = onnx.Constant dense<0> : tensor<i8>
  %limit = onnx.Constant dense<6> : tensor<i8>
  %delta = onnx.Constant dense<2> : tensor<i8>
  %result = "onnx.Range"(%start, %limit, %delta) : (tensor<i8>, tensor<i8>, tensor<i8>) -> tensor<3xi8>
  return %result : tensor<3xi8>
}

func.func @range_ui8() -> tensor<3xui8> {
  %start = onnx.Constant dense<0> : tensor<ui8>
  %limit = onnx.Constant dense<6> : tensor<ui8>
  %delta = onnx.Constant dense<2> : tensor<ui8>
  %result = "onnx.Range"(%start, %limit, %delta) : (tensor<ui8>, tensor<ui8>, tensor<ui8>) -> tensor<3xui8>
  return %result : tensor<3xui8>
}

func.func @range_i16() -> tensor<3xi16> {
  %start = onnx.Constant dense<0> : tensor<i16>
  %limit = onnx.Constant dense<6> : tensor<i16>
  %delta = onnx.Constant dense<2> : tensor<i16>
  %result = "onnx.Range"(%start, %limit, %delta) : (tensor<i16>, tensor<i16>, tensor<i16>) -> tensor<3xi16>
  return %result : tensor<3xi16>
}

func.func @range_ui16() -> tensor<3xui16> {
  %start = onnx.Constant dense<0> : tensor<ui16>
  %limit = onnx.Constant dense<6> : tensor<ui16>
  %delta = onnx.Constant dense<2> : tensor<ui16>
  %result = "onnx.Range"(%start, %limit, %delta) : (tensor<ui16>, tensor<ui16>, tensor<ui16>) -> tensor<3xui16>
  return %result : tensor<3xui16>
}

func.func @range_i32() -> tensor<3xi32> {
  %start = onnx.Constant dense<0> : tensor<i32>
  %limit = onnx.Constant dense<6> : tensor<i32>
  %delta = onnx.Constant dense<2> : tensor<i32>
  %result = "onnx.Range"(%start, %limit, %delta) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  return %result : tensor<3xi32>
}

func.func @range_ui32() -> tensor<3xui32> {
  %start = onnx.Constant dense<0> : tensor<ui32>
  %limit = onnx.Constant dense<6> : tensor<ui32>
  %delta = onnx.Constant dense<2> : tensor<ui32>
  %result = "onnx.Range"(%start, %limit, %delta) : (tensor<ui32>, tensor<ui32>, tensor<ui32>) -> tensor<3xui32>
  return %result : tensor<3xui32>
}

func.func @reduce_max_i8(%input: tensor<2x12xi8>) -> (tensor<2xi8>, tensor<2xi8>, tensor<2xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMax"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi8>, tensor<1xi64>) -> tensor<2xi8>
  %v18 = "onnx.ReduceMaxV18"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi8>, tensor<1xi64>) -> tensor<2xi8>
  %v13 = "onnx.ReduceMaxV13"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xi8>) -> tensor<2xi8>
  return %current, %v18, %v13 : tensor<2xi8>, tensor<2xi8>, tensor<2xi8>
}

func.func @reduce_max_ui8(%input: tensor<2x12xui8>) -> (tensor<2xui8>, tensor<2xui8>, tensor<2xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMax"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui8>, tensor<1xi64>) -> tensor<2xui8>
  %v18 = "onnx.ReduceMaxV18"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui8>, tensor<1xi64>) -> tensor<2xui8>
  %v13 = "onnx.ReduceMaxV13"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xui8>) -> tensor<2xui8>
  return %current, %v18, %v13 : tensor<2xui8>, tensor<2xui8>, tensor<2xui8>
}

func.func @reduce_max_i16(%input: tensor<2x12xi16>) -> (tensor<2xi16>, tensor<2xi16>, tensor<2xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMax"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi16>, tensor<1xi64>) -> tensor<2xi16>
  %v18 = "onnx.ReduceMaxV18"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi16>, tensor<1xi64>) -> tensor<2xi16>
  %v13 = "onnx.ReduceMaxV13"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xi16>) -> tensor<2xi16>
  return %current, %v18, %v13 : tensor<2xi16>, tensor<2xi16>, tensor<2xi16>
}

func.func @reduce_max_ui16(%input: tensor<2x12xui16>) -> (tensor<2xui16>, tensor<2xui16>, tensor<2xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMax"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui16>, tensor<1xi64>) -> tensor<2xui16>
  %v18 = "onnx.ReduceMaxV18"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui16>, tensor<1xi64>) -> tensor<2xui16>
  %v13 = "onnx.ReduceMaxV13"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xui16>) -> tensor<2xui16>
  return %current, %v18, %v13 : tensor<2xui16>, tensor<2xui16>, tensor<2xui16>
}

func.func @reduce_max_i32(%input: tensor<2x12xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMax"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi32>, tensor<1xi64>) -> tensor<2xi32>
  %v18 = "onnx.ReduceMaxV18"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi32>, tensor<1xi64>) -> tensor<2xi32>
  %v13 = "onnx.ReduceMaxV13"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xi32>) -> tensor<2xi32>
  return %current, %v18, %v13 : tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
}

func.func @reduce_max_ui32(%input: tensor<2x12xui32>) -> (tensor<2xui32>, tensor<2xui32>, tensor<2xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMax"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui32>, tensor<1xi64>) -> tensor<2xui32>
  %v18 = "onnx.ReduceMaxV18"(%input, %axes) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui32>, tensor<1xi64>) -> tensor<2xui32>
  %v13 = "onnx.ReduceMaxV13"(%input) {axes = [1], keepdims = 0 : si64} : (tensor<2x12xui32>) -> tensor<2xui32>
  return %current, %v18, %v13 : tensor<2xui32>, tensor<2xui32>, tensor<2xui32>
}

func.func @reduce_min_i8(%input: tensor<2x12xi8>) -> (tensor<2x1xi8>, tensor<2x1xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMin"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi8>, tensor<1xi64>) -> tensor<2x1xi8>
  %v18 = "onnx.ReduceMinV18"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi8>, tensor<1xi64>) -> tensor<2x1xi8>
  %v13 = "onnx.ReduceMinV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x12xi8>) -> tensor<2x1xi8>
  return %current, %v18, %v13 : tensor<2x1xi8>, tensor<2x1xi8>, tensor<2x1xi8>
}

func.func @reduce_min_ui8(%input: tensor<2x12xui8>) -> (tensor<2x1xui8>, tensor<2x1xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMin"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui8>, tensor<1xi64>) -> tensor<2x1xui8>
  %v18 = "onnx.ReduceMinV18"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui8>, tensor<1xi64>) -> tensor<2x1xui8>
  %v13 = "onnx.ReduceMinV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x12xui8>) -> tensor<2x1xui8>
  return %current, %v18, %v13 : tensor<2x1xui8>, tensor<2x1xui8>, tensor<2x1xui8>
}

func.func @reduce_min_i16(%input: tensor<2x12xi16>) -> (tensor<2x1xi16>, tensor<2x1xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMin"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi16>, tensor<1xi64>) -> tensor<2x1xi16>
  %v18 = "onnx.ReduceMinV18"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi16>, tensor<1xi64>) -> tensor<2x1xi16>
  %v13 = "onnx.ReduceMinV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x12xi16>) -> tensor<2x1xi16>
  return %current, %v18, %v13 : tensor<2x1xi16>, tensor<2x1xi16>, tensor<2x1xi16>
}

func.func @reduce_min_ui16(%input: tensor<2x12xui16>) -> (tensor<2x1xui16>, tensor<2x1xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMin"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui16>, tensor<1xi64>) -> tensor<2x1xui16>
  %v18 = "onnx.ReduceMinV18"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui16>, tensor<1xi64>) -> tensor<2x1xui16>
  %v13 = "onnx.ReduceMinV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x12xui16>) -> tensor<2x1xui16>
  return %current, %v18, %v13 : tensor<2x1xui16>, tensor<2x1xui16>, tensor<2x1xui16>
}

func.func @reduce_min_i32(%input: tensor<2x12xi32>) -> (tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMin"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi32>, tensor<1xi64>) -> tensor<2x1xi32>
  %v18 = "onnx.ReduceMinV18"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xi32>, tensor<1xi64>) -> tensor<2x1xi32>
  %v13 = "onnx.ReduceMinV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x12xi32>) -> tensor<2x1xi32>
  return %current, %v18, %v13 : tensor<2x1xi32>, tensor<2x1xi32>, tensor<2x1xi32>
}

func.func @reduce_min_ui32(%input: tensor<2x12xui32>) -> (tensor<2x1xui32>, tensor<2x1xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMin"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui32>, tensor<1xi64>) -> tensor<2x1xui32>
  %v18 = "onnx.ReduceMinV18"(%input, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x12xui32>, tensor<1xi64>) -> tensor<2x1xui32>
  %v13 = "onnx.ReduceMinV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x12xui32>) -> tensor<2x1xui32>
  return %current, %v18, %v13 : tensor<2x1xui32>, tensor<2x1xui32>, tensor<2x1xui32>
}

func.func @reduce_l1_i8(%input: tensor<2x3xi8>) -> (tensor<2xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL1"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceL1V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi8>) -> tensor<2x1xi8>
  return %current, %legacy : tensor<2xi8>, tensor<2x1xi8>
}

func.func @reduce_l1_ui8(%input: tensor<2x3xui8>) -> (tensor<2xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL1"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceL1V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui8>) -> tensor<2x1xui8>
  return %current, %legacy : tensor<2xui8>, tensor<2x1xui8>
}

func.func @reduce_l1_i16(%input: tensor<2x3xi16>) -> (tensor<2xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL1"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceL1V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi16>) -> tensor<2x1xi16>
  return %current, %legacy : tensor<2xi16>, tensor<2x1xi16>
}

func.func @reduce_l1_ui16(%input: tensor<2x3xui16>) -> (tensor<2xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL1"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceL1V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui16>) -> tensor<2x1xui16>
  return %current, %legacy : tensor<2xui16>, tensor<2x1xui16>
}

func.func @reduce_l1_i32(%input: tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL1"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceL1V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %current, %legacy : tensor<2xi32>, tensor<2x1xi32>
}

func.func @reduce_l1_ui32(%input: tensor<2x3xui32>) -> (tensor<2xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL1"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceL1V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui32>) -> tensor<2x1xui32>
  return %current, %legacy : tensor<2xui32>, tensor<2x1xui32>
}

func.func @reduce_l2_i8(%input: tensor<2x3xi8>) -> (tensor<2xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceL2V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi8>) -> tensor<2x1xi8>
  return %current, %legacy : tensor<2xi8>, tensor<2x1xi8>
}

func.func @reduce_l2_ui8(%input: tensor<2x3xui8>) -> (tensor<2xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceL2V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui8>) -> tensor<2x1xui8>
  return %current, %legacy : tensor<2xui8>, tensor<2x1xui8>
}

func.func @reduce_l2_i16(%input: tensor<2x3xi16>) -> (tensor<2xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceL2V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi16>) -> tensor<2x1xi16>
  return %current, %legacy : tensor<2xi16>, tensor<2x1xi16>
}

func.func @reduce_l2_ui16(%input: tensor<2x3xui16>) -> (tensor<2xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceL2V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui16>) -> tensor<2x1xui16>
  return %current, %legacy : tensor<2xui16>, tensor<2x1xui16>
}

func.func @reduce_l2_i32(%input: tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceL2V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %current, %legacy : tensor<2xi32>, tensor<2x1xi32>
}

func.func @reduce_l2_ui32(%input: tensor<2x3xui32>) -> (tensor<2xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceL2"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceL2V13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui32>) -> tensor<2x1xui32>
  return %current, %legacy : tensor<2xui32>, tensor<2x1xui32>
}

func.func @reduce_log_sum_i8(%input: tensor<2x3xi8>) -> (tensor<2xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceLogSumV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi8>) -> tensor<2x1xi8>
  return %current, %legacy : tensor<2xi8>, tensor<2x1xi8>
}

func.func @reduce_log_sum_ui8(%input: tensor<2x3xui8>) -> (tensor<2xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceLogSumV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui8>) -> tensor<2x1xui8>
  return %current, %legacy : tensor<2xui8>, tensor<2x1xui8>
}

func.func @reduce_log_sum_i16(%input: tensor<2x3xi16>) -> (tensor<2xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceLogSumV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi16>) -> tensor<2x1xi16>
  return %current, %legacy : tensor<2xi16>, tensor<2x1xi16>
}

func.func @reduce_log_sum_ui16(%input: tensor<2x3xui16>) -> (tensor<2xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceLogSumV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui16>) -> tensor<2x1xui16>
  return %current, %legacy : tensor<2xui16>, tensor<2x1xui16>
}

func.func @reduce_log_sum_i32(%input: tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceLogSumV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %current, %legacy : tensor<2xi32>, tensor<2x1xi32>
}

func.func @reduce_log_sum_ui32(%input: tensor<2x3xui32>) -> (tensor<2xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceLogSumV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui32>) -> tensor<2x1xui32>
  return %current, %legacy : tensor<2xui32>, tensor<2x1xui32>
}

func.func @reduce_log_sum_exp_i8(%input: tensor<2x3xi8>) -> (tensor<2xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSumExp"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceLogSumExpV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi8>) -> tensor<2x1xi8>
  return %current, %legacy : tensor<2xi8>, tensor<2x1xi8>
}

func.func @reduce_log_sum_exp_ui8(%input: tensor<2x3xui8>) -> (tensor<2xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSumExp"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceLogSumExpV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui8>) -> tensor<2x1xui8>
  return %current, %legacy : tensor<2xui8>, tensor<2x1xui8>
}

func.func @reduce_log_sum_exp_i16(%input: tensor<2x3xi16>) -> (tensor<2xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSumExp"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceLogSumExpV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi16>) -> tensor<2x1xi16>
  return %current, %legacy : tensor<2xi16>, tensor<2x1xi16>
}

func.func @reduce_log_sum_exp_ui16(%input: tensor<2x3xui16>) -> (tensor<2xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSumExp"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceLogSumExpV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui16>) -> tensor<2x1xui16>
  return %current, %legacy : tensor<2xui16>, tensor<2x1xui16>
}

func.func @reduce_log_sum_exp_i32(%input: tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSumExp"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceLogSumExpV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %current, %legacy : tensor<2xi32>, tensor<2x1xi32>
}

func.func @reduce_log_sum_exp_ui32(%input: tensor<2x3xui32>) -> (tensor<2xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceLogSumExp"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceLogSumExpV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui32>) -> tensor<2x1xui32>
  return %current, %legacy : tensor<2xui32>, tensor<2x1xui32>
}

func.func @reduce_mean_i8(%input: tensor<2x3xi8>) -> (tensor<2xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMean"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceMeanV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi8>) -> tensor<2x1xi8>
  return %current, %legacy : tensor<2xi8>, tensor<2x1xi8>
}

func.func @reduce_mean_ui8(%input: tensor<2x3xui8>) -> (tensor<2xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMean"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceMeanV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui8>) -> tensor<2x1xui8>
  return %current, %legacy : tensor<2xui8>, tensor<2x1xui8>
}

func.func @reduce_mean_i16(%input: tensor<2x3xi16>) -> (tensor<2xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMean"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceMeanV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi16>) -> tensor<2x1xi16>
  return %current, %legacy : tensor<2xi16>, tensor<2x1xi16>
}

func.func @reduce_mean_ui16(%input: tensor<2x3xui16>) -> (tensor<2xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMean"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceMeanV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui16>) -> tensor<2x1xui16>
  return %current, %legacy : tensor<2xui16>, tensor<2x1xui16>
}

func.func @reduce_mean_i32(%input: tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMean"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceMeanV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %current, %legacy : tensor<2xi32>, tensor<2x1xi32>
}

func.func @reduce_mean_ui32(%input: tensor<2x3xui32>) -> (tensor<2xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceMean"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceMeanV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui32>) -> tensor<2x1xui32>
  return %current, %legacy : tensor<2xui32>, tensor<2x1xui32>
}

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

func.func @reduce_sum_square_i8(%input: tensor<2x3xi8>) -> (tensor<2xi8>, tensor<2x1xi8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceSumSquare"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi8>, tensor<1xi64>) -> tensor<2xi8>
  %legacy = "onnx.ReduceSumSquareV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi8>) -> tensor<2x1xi8>
  return %current, %legacy : tensor<2xi8>, tensor<2x1xi8>
}

func.func @reduce_sum_square_ui8(%input: tensor<2x3xui8>) -> (tensor<2xui8>, tensor<2x1xui8>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceSumSquare"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui8>, tensor<1xi64>) -> tensor<2xui8>
  %legacy = "onnx.ReduceSumSquareV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui8>) -> tensor<2x1xui8>
  return %current, %legacy : tensor<2xui8>, tensor<2x1xui8>
}

func.func @reduce_sum_square_i16(%input: tensor<2x3xi16>) -> (tensor<2xi16>, tensor<2x1xi16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceSumSquare"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi16>, tensor<1xi64>) -> tensor<2xi16>
  %legacy = "onnx.ReduceSumSquareV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi16>) -> tensor<2x1xi16>
  return %current, %legacy : tensor<2xi16>, tensor<2x1xi16>
}

func.func @reduce_sum_square_ui16(%input: tensor<2x3xui16>) -> (tensor<2xui16>, tensor<2x1xui16>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceSumSquare"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui16>, tensor<1xi64>) -> tensor<2xui16>
  %legacy = "onnx.ReduceSumSquareV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui16>) -> tensor<2x1xui16>
  return %current, %legacy : tensor<2xui16>, tensor<2x1xui16>
}

func.func @reduce_sum_square_i32(%input: tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceSumSquare"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2xi32>
  %legacy = "onnx.ReduceSumSquareV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xi32>) -> tensor<2x1xi32>
  return %current, %legacy : tensor<2xi32>, tensor<2x1xi32>
}

func.func @reduce_sum_square_ui32(%input: tensor<2x3xui32>) -> (tensor<2xui32>, tensor<2x1xui32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %current = "onnx.ReduceSumSquare"(%input, %axes) {keepdims = 0 : si64} : (tensor<2x3xui32>, tensor<1xi64>) -> tensor<2xui32>
  %legacy = "onnx.ReduceSumSquareV13"(%input) {axes = [1], keepdims = 1 : si64} : (tensor<2x3xui32>) -> tensor<2x1xui32>
  return %current, %legacy : tensor<2xui32>, tensor<2x1xui32>
}

func.func @scaler_i8(%input: tensor<2x3xi8>) -> tensor<2x3xf32> {
  %result = "onnx.Scaler"(%input) {offset = [1.0 : f32], scale = [2.0 : f32]} : (tensor<2x3xi8>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

func.func @scaler_ui8(%input: tensor<2x3xui8>) -> tensor<2x3xf32> {
  %result = "onnx.Scaler"(%input) {offset = [1.0 : f32], scale = [2.0 : f32]} : (tensor<2x3xui8>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

func.func @scaler_i16(%input: tensor<2x3xi16>) -> tensor<2x3xf32> {
  %result = "onnx.Scaler"(%input) {offset = [1.0 : f32], scale = [2.0 : f32]} : (tensor<2x3xi16>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

func.func @scaler_ui16(%input: tensor<2x3xui16>) -> tensor<2x3xf32> {
  %result = "onnx.Scaler"(%input) {offset = [1.0 : f32], scale = [2.0 : f32]} : (tensor<2x3xui16>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

func.func @scaler_i32(%input: tensor<2x3xi32>) -> tensor<2x3xf32> {
  %result = "onnx.Scaler"(%input) {offset = [1.0 : f32], scale = [2.0 : f32]} : (tensor<2x3xi32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

func.func @scaler_ui32(%input: tensor<2x3xui32>) -> tensor<2x3xf32> {
  %result = "onnx.Scaler"(%input) {offset = [1.0 : f32], scale = [2.0 : f32]} : (tensor<2x3xui32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
