// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --split-input-file --batch-reduction-to-reshape-reduction %s | FileCheck %s
// Idempotence: a second application must not change the rewritten IR.
// RUN: onnx-mlir-opt --split-input-file --batch-reduction-to-reshape-reduction --batch-reduction-to-reshape-reduction %s | FileCheck %s

// ReduceSum over axis 1 of [B, R, 1, C] with B > 1 is rewritten as
// [1, B, R, C] -> ReduceSum(axes = [2], keepdims = 1) -> original shape.

// -----
// Static [8, 2, 1, 384], axis = 1, keepdims = 1.
func.func @batch8_axis1_keepdims1(%arg0: tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<8x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {ResultNames = ["rs_out"], keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<8x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<8x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @batch8_axis1_keepdims1
// CHECK-DAG:   %[[POST_SHAPE:.+]] = onnx.Constant dense<[8, 1, 1, 384]> : tensor<4xi64>
// CHECK-DAG:   %[[AXES:.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[PRE_SHAPE:.+]] = onnx.Constant dense<[1, 8, 2, 384]> : tensor<4xi64>
// CHECK:       %[[PRE:.+]] = "onnx.Reshape"(%arg0, %[[PRE_SHAPE]]) {allowzero = 0 : si64} : (tensor<8x2x1x384x!quant.uniform<u16:f32, 1.000000e-04:31222>>, tensor<4xi64>) -> tensor<1x8x2x384x!quant.uniform<u16:f32, 1.000000e-04:31222>>
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[PRE]], %[[AXES]]) {ResultNames = {{\[\[}}"rs_out", ["Reshape", [8, 1, 1, 384], [1, 8, 1, 384]]]], keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x2x384x!quant.uniform<u16:f32, 1.000000e-04:31222>>, tensor<1xi64>) -> tensor<1x8x1x384x!quant.uniform<u16:f32, 2.000000e-04:32629>>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[R]], %[[POST_SHAPE]]) {ResultNames = ["rs_out"], allowzero = 0 : si64} : (tensor<1x8x1x384x!quant.uniform<u16:f32, 2.000000e-04:32629>>, tensor<4xi64>) -> tensor<8x1x1x384x!quant.uniform<u16:f32, 2.000000e-04:32629>>
// CHECK:       return %[[POST]]
// CHECK-NOT:   "onnx.ReduceSum"

// -----
// Static [36, 2, 1, 384], negative axis = -3, keepdims = 0.
func.func @batch36_axis_neg3_keepdims0(%arg0: tensor<36x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<36x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<-3> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {ResultNames = ["rs_out"], keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<36x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<36x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<36x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @batch36_axis_neg3_keepdims0
// CHECK-DAG:   %[[POST_SHAPE:.+]] = onnx.Constant dense<[36, 1, 384]> : tensor<3xi64>
// CHECK-DAG:   %[[AXES:.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:   %[[PRE_SHAPE:.+]] = onnx.Constant dense<[1, 36, 2, 384]> : tensor<4xi64>
// CHECK:       %[[PRE:.+]] = "onnx.Reshape"(%arg0, %[[PRE_SHAPE]]) {allowzero = 0 : si64} : (tensor<36x2x1x384x!quant.uniform<u16:f32, 1.000000e-04:31222>>, tensor<4xi64>) -> tensor<1x36x2x384x!quant.uniform<u16:f32, 1.000000e-04:31222>>
// CHECK:       %[[R:.+]] = "onnx.ReduceSum"(%[[PRE]], %[[AXES]]) {ResultNames = {{\[\[}}"rs_out", ["Reshape", [36, 1, 384], [1, 36, 1, 384]]]], keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x36x2x384x!quant.uniform<u16:f32, 1.000000e-04:31222>>, tensor<1xi64>) -> tensor<1x36x1x384x!quant.uniform<u16:f32, 2.000000e-04:32629>>
// CHECK:       %[[POST:.+]] = "onnx.Reshape"(%[[R]], %[[POST_SHAPE]]) {ResultNames = ["rs_out"], allowzero = 0 : si64} : (tensor<1x36x1x384x!quant.uniform<u16:f32, 2.000000e-04:32629>>, tensor<3xi64>) -> tensor<36x1x384x!quant.uniform<u16:f32, 2.000000e-04:32629>>
// CHECK:       return %[[POST]]

// -----
// Dimension 2 is not singleton: not rewritten.
func.func @non_singleton_dim2_unchanged(%arg0: tensor<8x2x4x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<8x1x4x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<8x2x4x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<8x1x4x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<8x1x4x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @non_singleton_dim2_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"

// -----
// Unit batch: not rewritten.
func.func @unit_batch_unchanged(%arg0: tensor<1x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<1x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<1x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<1x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @unit_batch_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"

// -----
// Zero batch: not rewritten.
func.func @zero_batch_unchanged(%arg0: tensor<0x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<0x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<0x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<0x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<0x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @zero_batch_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"

// -----
// Dynamic batch: not rewritten.
func.func @dynamic_batch_unchanged(%arg0: tensor<?x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<?x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<?x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<?x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @dynamic_batch_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"

// -----
// Multiple axes: not rewritten.
func.func @multi_axis_unchanged(%arg0: tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<8x1x1x1x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<2xi64>) -> tensor<8x1x1x1x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<8x1x1x1x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @multi_axis_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"

// -----
// Runtime (non-constant) axes: not rewritten.
func.func @runtime_axes_unchanged(%arg0: tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, %axes: tensor<1xi64>) -> tensor<8x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<8x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<8x1x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @runtime_axes_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0, %arg1)
// CHECK-NOT:   "onnx.Reshape"

// -----
// Non-quantized input: not rewritten.
func.func @float_input_unchanged(%arg0: tensor<8x2x1x384xf32>) -> tensor<8x1x1x384xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<8x2x1x384xf32>, tensor<1xi64>) -> tensor<8x1x1x384xf32>
  return %0 : tensor<8x1x1x384xf32>
}
// CHECK-LABEL: func.func @float_input_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"

// -----
// Output shape inconsistent with keepdims = 1: not rewritten.
func.func @inconsistent_output_unchanged(%arg0: tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>) -> tensor<8x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<8x2x1x384x!quant.uniform<u16:f32, 1.0e-4:31222>>, tensor<1xi64>) -> tensor<8x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
  return %0 : tensor<8x1x384x!quant.uniform<u16:f32, 2.0e-4:32629>>
}
// CHECK-LABEL: func.func @inconsistent_output_unchanged
// CHECK-NOT:   "onnx.Reshape"
// CHECK:       "onnx.ReduceSum"(%arg0
// CHECK-NOT:   "onnx.Reshape"
