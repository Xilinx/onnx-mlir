// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference %s -split-input-file | FileCheck %s

// -----

// ReduceMeanV13 (attribute axes) is upgraded to the modern operand-axes
// ReduceMean. The axes attribute becomes a constant axes operand.
func.func @test_reducemeanv13_to_reducemean(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [1], keepdims = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
  onnx.Return %0 : tensor<2x1x4xf32>
// CHECK-LABEL:  func.func @test_reducemeanv13_to_reducemean
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[RES_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<2x1x4xf32>
// CHECK-NOT:       "onnx.ReduceMeanV13"
}

// -----

// A multi-axis attribute is preserved as a rank-N constant operand.
func.func @test_reducemeanv13_multi_axes(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x1x1x5xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [1, 2], keepdims = 1 : si64} : (tensor<2x3x4x5xf32>) -> tensor<2x1x1x5xf32>
  onnx.Return %0 : tensor<2x1x1x5xf32>
// CHECK-LABEL:  func.func @test_reducemeanv13_multi_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2x1x1x5xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK:           [[RES_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<2x1x1x5xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<2x1x1x5xf32>
// CHECK-NOT:       "onnx.ReduceMeanV13"
}

// -----

// Negative axes are carried through verbatim (not normalized to non-negative).
func.func @test_reducemeanv13_negative_axis(%arg0: tensor<1x384x768xf32>) -> tensor<1x384x1xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  onnx.Return %0 : tensor<1x384x1xf32>
// CHECK-LABEL:  func.func @test_reducemeanv13_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>) -> tensor<1x384x1xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK:           [[RES_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<1x384x1xf32>
// CHECK-NOT:       "onnx.ReduceMeanV13"
}

// -----

// keepdims = 0 is preserved through the upgrade.
func.func @test_reducemeanv13_keepdims_zero(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [1], keepdims = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
// CHECK-LABEL:  func.func @test_reducemeanv13_keepdims_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[RES_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[AXES_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<2x4xf32>
// CHECK-NOT:       "onnx.ReduceMeanV13"
}
