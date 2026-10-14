// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @fuse_reducesum_mul_single_axis
func.func @fuse_reducesum_mul_single_axis(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %scale = onnx.Constant dense<0.333333343> : tensor<f32>
  %sum = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %0 = "onnx.Mul"(%sum, %scale) : (tensor<2x1x4xf32>, tensor<f32>) -> tensor<2x1x4xf32>
  onnx.Return %0 : tensor<2x1x4xf32>
  // CHECK: "onnx.ReduceMean"(%arg0, %{{.*}}) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-NOT: "onnx.ReduceSum"
  // CHECK-NOT: "onnx.Mul"
}

// -----

// CHECK-LABEL: func.func @fuse_reducesum_mul_multi_axis
func.func @fuse_reducesum_mul_multi_axis(%arg0: tensor<1x3x4x5xf32>) -> tensor<1x1x1x5xf32> {
  %axes = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %scale = onnx.Constant dense<8.33333333E-2> : tensor<f32>
  %sum = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<1x3x4x5xf32>, tensor<2xi64>) -> tensor<1x1x1x5xf32>
  %0 = "onnx.Mul"(%scale, %sum) : (tensor<f32>, tensor<1x1x1x5xf32>) -> tensor<1x1x1x5xf32>
  onnx.Return %0 : tensor<1x1x1x5xf32>
  // CHECK: "onnx.ReduceMean"(%arg0, %{{.*}}) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-NOT: "onnx.ReduceSum"
  // CHECK-NOT: "onnx.Mul"
}

// -----

// MaterializeAbsentAxesReducePattern supplies the axes before this fusion.
// CHECK-LABEL: func.func @fuse_reducesum_mul_missing_axes
func.func @fuse_reducesum_mul_missing_axes(%arg0: tensor<2x3xf32>) -> tensor<1x1xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scale = onnx.Constant dense<0.166666672> : tensor<f32>
  %sum = "onnx.ReduceSum"(%arg0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3xf32>, none) -> tensor<1x1xf32>
  %0 = "onnx.Mul"(%sum, %scale) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  onnx.Return %0 : tensor<1x1xf32>
  // CHECK: "onnx.ReduceMean"(%arg0, %{{.*}}) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-NOT: "onnx.ReduceSum"
  // CHECK-NOT: "onnx.Mul"
}

// -----

// CHECK-LABEL: func.func @no_fuse_wrong_scale
func.func @no_fuse_wrong_scale(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %scale = onnx.Constant dense<0.5> : tensor<f32>
  %sum = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %0 = "onnx.Mul"(%sum, %scale) : (tensor<2x1x4xf32>, tensor<f32>) -> tensor<2x1x4xf32>
  onnx.Return %0 : tensor<2x1x4xf32>
  // CHECK: "onnx.ReduceSum"
  // CHECK: "onnx.Mul"
  // CHECK-NOT: "onnx.ReduceMean"
}

// -----

// An integer Mul by zero is not 1/N. isConstOf would truncate 1/N to 0.
// CHECK-LABEL: func.func @no_fuse_integer_zero_scale
func.func @no_fuse_integer_zero_scale(%arg0: tensor<2x3xi32>) -> tensor<2x1xi32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %scale = onnx.Constant dense<0> : tensor<i32>
  %sum = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3xi32>, tensor<1xi64>) -> tensor<2x1xi32>
  %0 = "onnx.Mul"(%sum, %scale) : (tensor<2x1xi32>, tensor<i32>) -> tensor<2x1xi32>
  onnx.Return %0 : tensor<2x1xi32>
  // CHECK: "onnx.ReduceSum"
  // CHECK: "onnx.Mul"
  // CHECK-NOT: "onnx.ReduceMean"
}

// -----

// The scale matches 1/N but broadcasts the reduced tensor to a wider shape.
// CHECK-LABEL: func.func @no_fuse_broadcasting_scale
func.func @no_fuse_broadcasting_scale(%arg0: tensor<2x3x4xf32>) -> tensor<2x5x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %scale = onnx.Constant dense<0.333333343> : tensor<2x5x4xf32>
  %sum = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %0 = "onnx.Mul"(%sum, %scale) : (tensor<2x1x4xf32>, tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
  onnx.Return %0 : tensor<2x5x4xf32>
  // CHECK: "onnx.ReduceSum"
  // CHECK: "onnx.Mul"
  // CHECK-SAME: -> tensor<2x5x4xf32>
  // CHECK-NOT: "onnx.ReduceMean"
}
