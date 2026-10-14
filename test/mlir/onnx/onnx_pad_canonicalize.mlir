// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// Zero padding is an identity for every Pad mode.
func.func @remove_zero_pad(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %pads = onnx.Constant dense<0> : tensor<6xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %result = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "reflect"} : (tensor<1x2x3xf32>, tensor<6xi64>, none, none) -> tensor<1x2x3xf32>
  return %result : tensor<1x2x3xf32>
}

// CHECK-LABEL: func.func @remove_zero_pad
// CHECK-NOT: onnx.Pad
// CHECK: return %arg0 : tensor<1x2x3xf32>

// -----

// A zero Pad whose result type is more precise than its input is kept, so the
// refined result type is not lost.
func.func @preserve_zero_pad_refining_type(%arg0: tensor<?x3xf32>) -> tensor<2x3xf32> {
  %pads = onnx.Constant dense<0> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %result = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"} : (tensor<?x3xf32>, tensor<4xi64>, none, none) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @preserve_zero_pad_refining_type
// CHECK: "onnx.Pad"(%arg0,
// CHECK-SAME: -> tensor<2x3xf32>

// -----

// Equal input and output shapes do not make a nonzero Pad redundant.
// This Pad crops one element and adds one element.
func.func @preserve_same_shape_nonzero_pad(
    %arg0: tensor<4xf32>) -> tensor<4xf32> {
  %pads = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %result = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"} : (tensor<4xf32>, tensor<2xi64>, none, none) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @preserve_same_shape_nonzero_pad
// CHECK: onnx.Pad
// CHECK: return

// -----

// Consecutive constant-mode Pads with the same default fill and nonnegative
// full-rank pads are fused by adding their pad amounts.
func.func @fuse_constant_pads(%arg0: tensor<2x3xf32>) -> tensor<6x7xf32> {
  %inner_pads = onnx.Constant dense<[0, 1, 1, 2]> : tensor<4xi64>
  %outer_pads = onnx.Constant dense<[1, 0, 2, 1]> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %none, %none) {mode = "constant"} : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<3x6xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %none, %none) {mode = "constant"} : (tensor<3x6xf32>, tensor<4xi64>, none, none) -> tensor<6x7xf32>
  return %outer : tensor<6x7xf32>
}

// CHECK-LABEL: func.func @fuse_constant_pads
// CHECK-DAG: %[[PADS:.*]] = onnx.Constant dense<[1, 1, 3, 3]> : tensor<4xi64>
// CHECK-DAG: %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]], %[[NONE]], %[[NONE]]) {mode = "constant"} : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<6x7xf32>
// CHECK-NOT: onnx.Pad
// CHECK: return %[[PAD]] : tensor<6x7xf32>

// -----

// Equal explicit constant fill values are also fusible.
func.func @fuse_constant_pads_same_explicit_fill(
    %arg0: tensor<2x3xf32>) -> tensor<6x7xf32> {
  %inner_pads = onnx.Constant dense<[0, 1, 1, 2]> : tensor<4xi64>
  %outer_pads = onnx.Constant dense<[1, 0, 2, 1]> : tensor<4xi64>
  %inner_value = onnx.Constant dense<3.0> : tensor<f32>
  %outer_value = onnx.Constant dense<3.0> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %inner_value, %none) {
    mode = "constant"
  } : (tensor<2x3xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<3x6xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %outer_value, %none) {
    mode = "constant"
  } : (tensor<3x6xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<6x7xf32>
  return %outer : tensor<6x7xf32>
}

// CHECK-LABEL: func.func @fuse_constant_pads_same_explicit_fill
// CHECK-DAG: %[[PADS:.*]] = onnx.Constant dense<[1, 1, 3, 3]> : tensor<4xi64>
// CHECK-DAG: %[[FILL:.*]] = onnx.Constant dense<3.000000e+00> : tensor<f32>
// CHECK-DAG: %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]], %[[FILL]], %[[NONE]]) {mode = "constant"} : (tensor<2x3xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<6x7xf32>
// CHECK-NOT: onnx.Pad
// CHECK: return %[[PAD]] : tensor<6x7xf32>

// -----

// Non-constant modes are deliberately not fused.
func.func @preserve_reflect_pad_chain(%arg0: tensor<4xf32>) -> tensor<8xf32> {
  %inner_pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %outer_pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %none, %none) {mode = "reflect"} : (tensor<4xf32>, tensor<2xi64>, none, none) -> tensor<6xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %none, %none) {mode = "reflect"} : (tensor<6xf32>, tensor<2xi64>, none, none) -> tensor<8xf32>
  return %outer : tensor<8xf32>
}

// CHECK-LABEL: func.func @preserve_reflect_pad_chain
// CHECK-COUNT-2: onnx.Pad
// CHECK-NOT: onnx.Pad
// CHECK: return

// -----

// Different constant fill values are not equivalent to one Pad.
func.func @preserve_different_constant_fills(
    %arg0: tensor<4xf32>) -> tensor<8xf32> {
  %inner_pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %outer_pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %inner_value = onnx.Constant dense<1.0> : tensor<f32>
  %outer_value = onnx.Constant dense<2.0> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %inner_value, %none) {
    mode = "constant"
  } : (tensor<4xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<6xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %outer_value, %none) {
    mode = "constant"
  } : (tensor<6xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<8xf32>
  return %outer : tensor<8xf32>
}

// CHECK-LABEL: func.func @preserve_different_constant_fills
// CHECK-COUNT-2: onnx.Pad
// CHECK-NOT: onnx.Pad
// CHECK: return

// -----

// Cropping and then padding is not equivalent to adding the pad amounts: the
// inner Pad drops a data element that the outer Pad replaces with the fill.
// Fusing would produce all-zero pads and silently remove both Pads.
func.func @preserve_negative_pad_chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %inner_pads = onnx.Constant dense<[-1, 0]> : tensor<2xi64>
  %outer_pads = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %none, %none) {mode = "constant"} : (tensor<4xf32>, tensor<2xi64>, none, none) -> tensor<3xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %none, %none) {mode = "constant"} : (tensor<3xf32>, tensor<2xi64>, none, none) -> tensor<4xf32>
  return %outer : tensor<4xf32>
}

// CHECK-LABEL: func.func @preserve_negative_pad_chain
// CHECK-COUNT-2: onnx.Pad
// CHECK-NOT: onnx.Pad
// CHECK: return

// -----

// Pads with the same explicit axis are normalized and fused.
func.func @fuse_pad_chain_with_axes(
    %arg0: tensor<2x4xf32>) -> tensor<2x8xf32> {
  %pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x4xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<2x6xf32>
  %outer = "onnx.Pad"(%inner, %pads, %none, %axes) {mode = "constant"} : (tensor<2x6xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<2x8xf32>
  return %outer : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @fuse_pad_chain_with_axes
// CHECK-DAG: %[[PADS:.*]] = onnx.Constant dense<[0, 2, 0, 2]> : tensor<4xi64>
// CHECK-DAG: %[[AXES:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]], {{.*}}, %[[AXES]]) {mode = "constant"} : (tensor<2x4xf32>, tensor<4xi64>, none, none) -> tensor<2x8xf32>
// CHECK-NOT: onnx.Pad
// CHECK: return %[[PAD]] : tensor<2x8xf32>

// -----

// Pads acting on different axes are also fused after normalization.
func.func @fuse_pad_chain_on_different_axes(
    %arg0: tensor<2x3xf32>) -> tensor<5x10xf32> {
  %inner_pads = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %outer_pads = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %inner_axes = onnx.Constant dense<0> : tensor<1xi64>
  %outer_axes = onnx.Constant dense<1> : tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %none, %inner_axes) {
    mode = "constant"
  } : (tensor<2x3xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<5x3xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %none, %outer_axes) {
    mode = "constant"
  } : (tensor<5x3xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<5x10xf32>
  return %outer : tensor<5x10xf32>
}

// CHECK-LABEL: func.func @fuse_pad_chain_on_different_axes
// CHECK-DAG: %[[PADS:.*]] = onnx.Constant dense<[1, 3, 2, 4]> : tensor<4xi64>
// CHECK-DAG: %[[AXES:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]], {{.*}}, %[[AXES]]) {mode = "constant"} : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<5x10xf32>
// CHECK-NOT: onnx.Pad
// CHECK: return %[[PAD]] : tensor<5x10xf32>

// -----

// A Pad with explicit axes is fused with a full-rank Pad without axes.
func.func @fuse_pad_chain_mixed_axes(
    %arg0: tensor<2x3xf32>) -> tensor<4x6xf32> {
  %inner_pads = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %outer_pads = onnx.Constant dense<[1, 0, 1, 0]> : tensor<4xi64>
  %inner_axes = onnx.Constant dense<1> : tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %none, %inner_axes) {
    mode = "constant"
  } : (tensor<2x3xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<2x6xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %none, %none) {
    mode = "constant"
  } : (tensor<2x6xf32>, tensor<4xi64>, none, none) -> tensor<4x6xf32>
  return %outer : tensor<4x6xf32>
}

// CHECK-LABEL: func.func @fuse_pad_chain_mixed_axes
// CHECK-DAG: %[[PADS:.*]] = onnx.Constant dense<[1, 1, 1, 2]> : tensor<4xi64>
// CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]], {{.*}}) {mode = "constant"} : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<4x6xf32>
// CHECK-NOT: onnx.Pad
// CHECK: return %[[PAD]] : tensor<4x6xf32>

// -----

// Negative axes are normalized before fusing.
func.func @fuse_pad_chain_with_negative_axis(
    %arg0: tensor<2x4xf32>) -> tensor<7x6xf32> {
  %inner_pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %outer_pads = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %inner_axes = onnx.Constant dense<-1> : tensor<1xi64>
  %outer_axes = onnx.Constant dense<-2> : tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %inner = "onnx.Pad"(%arg0, %inner_pads, %none, %inner_axes) {
    mode = "constant"
  } : (tensor<2x4xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<2x6xf32>
  %outer = "onnx.Pad"(%inner, %outer_pads, %none, %outer_axes) {
    mode = "constant"
  } : (tensor<2x6xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<7x6xf32>
  return %outer : tensor<7x6xf32>
}

// CHECK-LABEL: func.func @fuse_pad_chain_with_negative_axis
// CHECK-DAG: %[[PADS:.*]] = onnx.Constant dense<[2, 1, 3, 1]> : tensor<4xi64>
// CHECK-DAG: %[[AXES:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]], {{.*}}, %[[AXES]]) {mode = "constant"} : (tensor<2x4xf32>, tensor<4xi64>, none, none) -> tensor<7x6xf32>
// CHECK-NOT: onnx.Pad
// CHECK: return %[[PAD]] : tensor<7x6xf32>
