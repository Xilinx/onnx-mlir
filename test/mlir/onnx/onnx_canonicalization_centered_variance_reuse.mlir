// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --canonicalize-with-rn %s --split-input-file | FileCheck %s

// Reuse an earlier centered subtraction while preserving its shared users.
func.func @earlier(%x: tensor<2x4xf32>) -> (tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  %centered = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %clamp = "onnx.Max"(%variance, %zero) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  return %clamp, %centered, %centered : tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}

// CHECK-LABEL:  func.func @earlier(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x4xf32>) -> (tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
// CHECK:           return [[VAR_4_]], [[VAR_2_]], [[VAR_2_]] : tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>
// CHECK:         }

// -----

// Move a later centered subtraction before the variance and preserve its users.
func.func @later(%x: tensor<2x4xf32>) -> (tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  %clamp = "onnx.Max"(%variance, %zero) : (tensor<2x1xf32>, tensor<f32>) -> tensor<2x1xf32>
  %centered = "onnx.Sub"(%x, %mean) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
  return %clamp, %centered, %centered : tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}

// CHECK-LABEL:  func.func @later(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x4xf32>) -> (tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<2x1xf32>
// CHECK:           return [[VAR_4_]], [[VAR_2_]], [[VAR_2_]] : tensor<2x1xf32>, tensor<2x4xf32>, tensor<2x4xf32>
// CHECK:         }
