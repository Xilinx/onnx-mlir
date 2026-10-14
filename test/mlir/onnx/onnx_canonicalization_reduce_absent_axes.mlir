// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference %s -split-input-file | FileCheck %s

// -----

func.func @reduce_l1_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceL1"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_l1_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceL1"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}

// -----

func.func @reduce_l2_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceL2"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_l2_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceL2"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}

// -----

func.func @reduce_max_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMax"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_max_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceMax"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}

// -----

func.func @reduce_mean_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMean"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_mean_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceMean"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}

// -----

func.func @reduce_min_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceMin"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_min_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceMin"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}

// -----

func.func @reduce_prod_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceProd"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_prod_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceProd"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}

// -----

func.func @reduce_sum_absent_axes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceSum"(%arg0, %none) {keepdims = 1 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL: func.func @reduce_sum_absent_axes
// CHECK: [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK: "onnx.ReduceSum"(%arg0, [[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
}
