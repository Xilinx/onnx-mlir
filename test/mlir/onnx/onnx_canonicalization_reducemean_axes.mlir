// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference %s -split-input-file | FileCheck %s

// -----

// Modern ReduceMean with an absent (None) axes operand and default semantics
// (noop_with_empty_axes = 0) means "reduce all dims"; the materialization
// pattern makes the axes explicit as [0, 1, ..., rank-1].
func.func @test_reducemean_noaxes_materialize(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMean"(%arg0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL:  func.func @test_reducemean_noaxes_materialize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK:           [[RES_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<1x1x1xf32>
// CHECK-NOT:       "onnx.NoValue"
}

// -----

// ReduceMeanV13 without an axes attribute is first upgraded to a modern
// ReduceMean with a None axes operand, which is then materialized to an
// explicit [0, 1, ..., rank-1] by the materialization pattern.
func.func @test_reducemeanv13_noaxes_materialize(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {keepdims = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL:  func.func @test_reducemeanv13_noaxes_materialize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// CHECK:           [[AXES_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK:           [[RES_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[AXES_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<1x1x1xf32>
// CHECK-NOT:       "onnx.ReduceMeanV13"
// CHECK-NOT:       "onnx.NoValue"
}

// -----

// Modern ReduceMean with an absent (None) axes operand and
// noop_with_empty_axes = 1 is a genuine no-op that forwards `data`; the
// materialization pattern must NOT fire here.
func.func @test_reducemean_noaxes_noop_not_materialized(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMean"(%arg0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<2x3x4xf32>
  onnx.Return %0 : tensor<2x3x4xf32>
// CHECK-LABEL:  func.func @test_reducemean_noaxes_noop_not_materialized
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK:           onnx.Return [[PARAM_0_]] : tensor<2x3x4xf32>
// CHECK-NOT:       "onnx.Constant"
}
