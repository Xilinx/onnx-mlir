// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// RUN: onnx-mlir-opt --fuse-reshape-pairs %s | FileCheck %s

func.func @fold_cancelling_reshape_pair(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %flat = onnx.Constant dense<[1, 32]> : tensor<2xi64>
  %orig = onnx.Constant dense<[1, 4, 8]> : tensor<3xi64>

  %0 = "onnx.Relu"(%arg0) {ResultNames = ["relu_out"]} : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %1 = "onnx.Reshape"(%0, %flat) {ResultNames = ["reshape0_out"], allowzero = 0 : si64} : (tensor<1x4x8xf32>, tensor<2xi64>) -> tensor<1x32xf32>
  %2 = "onnx.Reshape"(%1, %orig) {ResultNames = ["reshape1_out"], allowzero = 0 : si64} : (tensor<1x32xf32>, tensor<3xi64>) -> tensor<1x4x8xf32>
  return %2 : tensor<1x4x8xf32>
}
// CHECK-LABEL: @fold_cancelling_reshape_pair
// CHECK: "onnx.Relu"
// CHECK-SAME: ResultNames = ["reshape1_out"]
// CHECK-NOT: "onnx.Reshape"

// A Squeeze undoing an Unsqueeze is the same fold across op types.
func.func @fold_cancelling_unsqueeze_squeeze(%arg0: tensor<1x38x46x46xf32>) -> tensor<1x38x46x46xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>

  %0 = "onnx.Relu"(%arg0) {ResultNames = ["relu_out"]} : (tensor<1x38x46x46xf32>) -> tensor<1x38x46x46xf32>
  %1 = "onnx.Unsqueeze"(%0, %axes) {ResultNames = ["unsqueeze_out"]} : (tensor<1x38x46x46xf32>, tensor<1xi64>) -> tensor<1x1x38x46x46xf32>
  %2 = "onnx.Squeeze"(%1, %axes) {ResultNames = ["squeeze_out"]} : (tensor<1x1x38x46x46xf32>, tensor<1xi64>) -> tensor<1x38x46x46xf32>
  return %2 : tensor<1x38x46x46xf32>
}
// CHECK-LABEL: @fold_cancelling_unsqueeze_squeeze
// CHECK: "onnx.Relu"
// CHECK-SAME: ResultNames = ["squeeze_out"]
// CHECK-NOT: "onnx.Unsqueeze"
// CHECK-NOT: "onnx.Squeeze"

// Non-cancelling neighbours are left alone; merging them into a single Reshape
// would hide the shape change from ONNXTransposeOptimizationPass.
func.func @keep_non_cancelling_reshape_pair(%arg0: tensor<1x4x8xf32>) -> tensor<2x16xf32> {
  %flat = onnx.Constant dense<[1, 32]> : tensor<2xi64>
  %pair = onnx.Constant dense<[2, 16]> : tensor<2xi64>

  %0 = "onnx.Relu"(%arg0) {ResultNames = ["relu_out"]} : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %1 = "onnx.Reshape"(%0, %flat) {ResultNames = ["reshape0_out"], allowzero = 0 : si64} : (tensor<1x4x8xf32>, tensor<2xi64>) -> tensor<1x32xf32>
  %2 = "onnx.Reshape"(%1, %pair) {ResultNames = ["reshape1_out"], allowzero = 0 : si64} : (tensor<1x32xf32>, tensor<2xi64>) -> tensor<2x16xf32>
  return %2 : tensor<2x16xf32>
}
// CHECK-LABEL: @keep_non_cancelling_reshape_pair
// CHECK: "onnx.Reshape"
// CHECK-SAME: ResultNames = ["reshape0_out"]
// CHECK: "onnx.Reshape"
// CHECK-SAME: ResultNames = ["reshape1_out"]

// The attention pattern from PSO3: an Unsqueeze that cancels nothing must stay
// an Unsqueeze, otherwise ONNXTransposeOptimizationPass can no longer rewrite
// the permutation of the transpose that consumes it.
func.func @keep_unsqueeze_that_does_not_cancel(%arg0: tensor<151x1x2304xf32>) -> tensor<1x151x1x3x768xf32> {
  %shape = onnx.Constant dense<[151, 1, 3, 768]> : tensor<4xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>

  %0 = "onnx.Reshape"(%arg0, %shape) {ResultNames = ["qkv_reshape"], allowzero = 0 : si64} : (tensor<151x1x2304xf32>, tensor<4xi64>) -> tensor<151x1x3x768xf32>
  %1 = "onnx.Unsqueeze"(%0, %axes) {ResultNames = ["qkv_unsqueeze"]} : (tensor<151x1x3x768xf32>, tensor<1xi64>) -> tensor<1x151x1x3x768xf32>
  return %1 : tensor<1x151x1x3x768xf32>
}
// CHECK-LABEL: @keep_unsqueeze_that_does_not_cancel
// CHECK: "onnx.Reshape"
// CHECK-SAME: ResultNames = ["qkv_reshape"]
// CHECK: "onnx.Unsqueeze"
// CHECK-SAME: ResultNames = ["qkv_unsqueeze"]

// The intermediate is consumed elsewhere, so removing the pair would change
// that other consumer's input.
func.func @keep_pair_when_intermediate_is_shared(%arg0: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x32xf32>) {
  %flat = onnx.Constant dense<[1, 32]> : tensor<2xi64>
  %orig = onnx.Constant dense<[1, 4, 8]> : tensor<3xi64>

  %0 = "onnx.Relu"(%arg0) {ResultNames = ["relu_out"]} : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %1 = "onnx.Reshape"(%0, %flat) {ResultNames = ["reshape0_out"], allowzero = 0 : si64} : (tensor<1x4x8xf32>, tensor<2xi64>) -> tensor<1x32xf32>
  %2 = "onnx.Reshape"(%1, %orig) {ResultNames = ["reshape1_out"], allowzero = 0 : si64} : (tensor<1x32xf32>, tensor<3xi64>) -> tensor<1x4x8xf32>
  return %2, %1 : tensor<1x4x8xf32>, tensor<1x32xf32>
}
// CHECK-LABEL: @keep_pair_when_intermediate_is_shared
// CHECK: "onnx.Relu"
// CHECK-SAME: ResultNames = ["relu_out"]
// CHECK: "onnx.Reshape"
// CHECK-SAME: ResultNames = ["reshape0_out"]
// CHECK: "onnx.Reshape"
// CHECK-SAME: ResultNames = ["reshape1_out"]
