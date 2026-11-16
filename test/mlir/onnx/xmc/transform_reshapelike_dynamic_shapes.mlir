// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --transform-reshapelike-op-to-reshape %s | FileCheck %s
// RUN: onnx-mlir-opt --split-input-file --transform-reshapelike-op-to-reshape --canonicalize %s | FileCheck %s

// NMS can produce a runtime number of labels even when the model inputs have
// static shapes. Converting this Unsqueeze to Reshape used to put INT64_MIN in
// a constant shape tensor and abort during the following Add canonicalization.
func.func @dynamic_unsqueeze_add(%labels: tensor<?xi64>, %bias: tensor<i64>) -> tensor<1x?xi64> {
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %expanded = "onnx.Unsqueeze"(%labels, %axes) : (tensor<?xi64>, tensor<1xi64>) -> tensor<1x?xi64>
  %result = "onnx.Add"(%expanded, %bias) : (tensor<1x?xi64>, tensor<i64>) -> tensor<1x?xi64>
  return %result : tensor<1x?xi64>
}
// CHECK-LABEL: func.func @dynamic_unsqueeze_add
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.Unsqueeze"
// CHECK: "onnx.Add"
// CHECK-NOT: onnx.Reshape
// CHECK: return

// -----

func.func @dynamic_squeeze(%arg0: tensor<1x?x3xf32>) -> tensor<?x3xf32> {
  %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %result = "onnx.Squeeze"(%arg0, %axes) : (tensor<1x?x3xf32>, tensor<1xi64>) -> tensor<?x3xf32>
  return %result : tensor<?x3xf32>
}
// CHECK-LABEL: func.func @dynamic_squeeze
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.Squeeze"
// CHECK-NOT: onnx.Reshape
// CHECK: return

// -----

func.func @dynamic_flatten(%arg0: tensor<?x3x4xf32>) -> tensor<?x12xf32> {
  %result = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<?x3x4xf32>) -> tensor<?x12xf32>
  return %result : tensor<?x12xf32>
}
// CHECK-LABEL: func.func @dynamic_flatten
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.Flatten"
// CHECK-NOT: onnx.Reshape
// CHECK: return

// -----

// Unknown dimensions must not be treated as singleton dimensions: this
// transpose reorders data when both dimensions have runtime sizes above one.
func.func @dynamic_transpose(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @dynamic_transpose
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.Transpose"
// CHECK-NOT: onnx.Reshape
// CHECK: return

// -----

// Even a transpose that only moves a singleton axis needs a runtime shape.
func.func @dynamic_trivial_transpose(%arg0: tensor<?x1xf32>) -> tensor<1x?xf32> {
  %result = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<?x1xf32>) -> tensor<1x?xf32>
  return %result : tensor<1x?xf32>
}
// CHECK-LABEL: func.func @dynamic_trivial_transpose
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.Transpose"
// CHECK-NOT: onnx.Reshape
// CHECK: return
