// RUN: onnx-mlir-opt --split-input-file --replace-qdq-reduce-l2 %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Tests for ReplaceQDQReduceL2Pass: fuse the quantized L2-norm
// subgraph Square->ReduceSum->[eps-Add]->Sqrt into onnx.ReduceL2. The pass runs
// after QuantTypesPass, so inputs use native !quant.uniform types (no explicit
// Q/DQ ops). The square is Mul(x,x) (Pow(x,2) is canonicalized to Mul before
// the xmc passes) or Pow(x,2).

// -----
// Pattern B, square as Mul(x, x) (the real post-quant-types form).
// CHECK-LABEL: func.func @l2norm_mul_square
// CHECK-DAG:   %[[AXES:.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:       "onnx.ReduceL2"(%arg0, %[[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK-NOT:   "onnx.Mul"
// CHECK-NOT:   "onnx.Sqrt"
func.func @l2norm_mul_square(
    %arg0: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>)
    -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %sq = "onnx.Mul"(%arg0, %arg0) : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %rs = "onnx.ReduceSum"(%sq, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1xi64>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %sqrt = "onnx.Sqrt"(%rs) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  return %sqrt : tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----
// Pattern B, square as Pow(x, 2).
// CHECK-LABEL: func.func @l2norm_pow_square
// CHECK:       "onnx.ReduceL2"(%arg0,
// CHECK-NOT:   "onnx.Pow"
// CHECK-NOT:   "onnx.Sqrt"
func.func @l2norm_pow_square(
    %arg0: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>)
    -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>> {
  %two = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %sq = "onnx.Pow"(%arg0, %two) : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<f32>) -> tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %rs = "onnx.ReduceSum"(%sq, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1xi64>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %sqrt = "onnx.Sqrt"(%rs) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  return %sqrt : tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----
// Pattern A: eps=0 Add with a native quant-typed zero constant (storage == zp).
// CHECK-LABEL: func.func @l2norm_eps_add
// CHECK:       "onnx.ReduceL2"(%arg0,
// CHECK-NOT:   "onnx.Mul"
// CHECK-NOT:   "onnx.Add"
// CHECK-NOT:   "onnx.Sqrt"
func.func @l2norm_eps_add(
    %arg0: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>)
    -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %eps = onnx.Constant {value = dense<0> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 1.000000e+00>>
  %sq = "onnx.Mul"(%arg0, %arg0) : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %rs = "onnx.ReduceSum"(%sq, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1xi64>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %add = "onnx.Add"(%rs, %eps) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %sqrt = "onnx.Sqrt"(%add) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  return %sqrt : tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----
// Negative: eps-Add whose output quant type differs from its input (a real
// requantization, not a transparent no-op) must NOT match.
// CHECK-LABEL: func.func @l2norm_eps_add_requant
// CHECK:       "onnx.Add"
// CHECK:       "onnx.Sqrt"
// CHECK-NOT:   "onnx.ReduceL2"
func.func @l2norm_eps_add_requant(
    %arg0: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>)
    -> tensor<1x1x96x224x!quant.uniform<i8:f32, 2.000000e+00>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %eps = onnx.Constant {value = dense<0> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 1.000000e+00>>
  %sq = "onnx.Mul"(%arg0, %arg0) : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %rs = "onnx.ReduceSum"(%sq, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1xi64>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %add = "onnx.Add"(%rs, %eps) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 2.000000e+00>>
  %sqrt = "onnx.Sqrt"(%add) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 2.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 2.000000e+00>>
  return %sqrt : tensor<1x1x96x224x!quant.uniform<i8:f32, 2.000000e+00>>
}

// -----
// Negative: Mul with distinct operands (not a square) must NOT match.
// CHECK-LABEL: func.func @not_a_square_mul
// CHECK:       "onnx.Mul"
// CHECK-NOT:   "onnx.ReduceL2"
func.func @not_a_square_mul(
    %arg0: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>,
    %arg1: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>)
    -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %mul = "onnx.Mul"(%arg0, %arg1) : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %rs = "onnx.ReduceSum"(%mul, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1xi64>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %sqrt = "onnx.Sqrt"(%rs) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  return %sqrt : tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----
// Negative: Pow exponent != 2 must NOT match.
// CHECK-LABEL: func.func @l2norm_wrong_exponent
// CHECK:       "onnx.Pow"
// CHECK-NOT:   "onnx.ReduceL2"
func.func @l2norm_wrong_exponent(
    %arg0: tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>)
    -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>> {
  %three = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %sq = "onnx.Pow"(%arg0, %three) : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<f32>) -> tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %rs = "onnx.ReduceSum"(%sq, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x96x224x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<1xi64>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  %sqrt = "onnx.Sqrt"(%rs) : (tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
  return %sqrt : tensor<1x1x96x224x!quant.uniform<i8:f32, 1.000000e+00>>
}
