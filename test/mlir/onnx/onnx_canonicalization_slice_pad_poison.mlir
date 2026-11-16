// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt %s --canonicalize -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt %s --shape-inference --canonicalize -split-input-file | FileCheck %s

// -----

func.func @cancel_low(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[3]> : tensor<1xi64>
  %ends = onnx.Constant dense<[10]> : tensor<1xi64>
  %pads = onnx.Constant dense<[3, 0]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<7xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<7xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @cancel_low(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @cancel_high(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[0]> : tensor<1xi64>
  %ends = onnx.Constant dense<[7]> : tensor<1xi64>
  %pads = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<7xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<7xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @cancel_high(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @cancel_both(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @cancel_both(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @cancel_multi_axis(%x: tensor<8x10xf32>) -> (tensor<8x10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %ends = onnx.Constant dense<[7, 8]> : tensor<2xi64>
  %pads = onnx.Constant dense<[1, 3, 1, 2]> : tensor<4xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<8x10xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<6x5xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6x5xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<8x10xf32>
  return %result : tensor<8x10xf32>
}

// CHECK-LABEL: func.func @cancel_multi_axis(
// CHECK-SAME: [[X:%.+]]: tensor<8x10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @backfill_to_slice(%x: tensor<12xf32>) -> (tensor<6xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[3]> : tensor<1xi64>
  %ends = onnx.Constant dense<[7]> : tensor<1xi64>
  %pads = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<4xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<4xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// CHECK-LABEL: func.func @backfill_to_slice(
// CHECK-SAME: [[X:%.+]]: tensor<12xf32>
// CHECK-DAG: [[START:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG: [[END:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]], [[START]], [[END]],
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[SLICE]]

// -----

func.func @backfill_beyond_input(%x: tensor<10xf32>) -> (tensor<12xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[3]> : tensor<1xi64>
  %ends = onnx.Constant dense<[7]> : tensor<1xi64>
  %pads = onnx.Constant dense<[4, 4]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<4xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<4xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<12xf32>
  return %result : tensor<12xf32>
}

// CHECK-LABEL: func.func @backfill_beyond_input(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK-DAG: [[PADS:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK-DAG: [[POISON:%.+]] = ub.poison : tensor<f32>
// CHECK-NOT: "onnx.Slice"
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[X]], [[PADS]], [[POISON]],
// CHECK: return [[RESULT]]

// -----

func.func @widen_slice_keep_pad(%x: tensor<12xf32>) -> (tensor<14xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[5]> : tensor<1xi64>
  %ends = onnx.Constant dense<[9]> : tensor<1xi64>
  %pads = onnx.Constant dense<[10, 0]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<4xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<4xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<14xf32>
  return %result : tensor<14xf32>
}

// CHECK-LABEL: func.func @widen_slice_keep_pad(
// CHECK-SAME: [[X:%.+]]: tensor<12xf32>
// CHECK-DAG: [[START:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG: [[END:%.+]] = onnx.Constant dense<9> : tensor<1xi64>
// CHECK-DAG: [[PADS:%.+]] = onnx.Constant dense<[5, 0]> : tensor<2xi64>
// CHECK-DAG: [[POISON:%.+]] = ub.poison : tensor<f32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]], [[START]], [[END]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]], [[PADS]], [[POISON]],
// CHECK: return [[RESULT]]

// -----

func.func @multi_axis_partial(%x: tensor<8x8xf32>) -> (tensor<6x7xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %ends = onnx.Constant dense<[6, 6]> : tensor<2xi64>
  %pads = onnx.Constant dense<[1, 3, 1, 0]> : tensor<4xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<8x8xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<4x4xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<4x4xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<6x7xf32>
  return %result : tensor<6x7xf32>
}

// CHECK-LABEL: func.func @multi_axis_partial(
// CHECK-SAME: [[X:%.+]]: tensor<8x8xf32>
// CHECK-DAG: [[START:%.+]] = onnx.Constant dense<[1, 0]> : tensor<2xi64>
// CHECK-DAG: [[END:%.+]] = onnx.Constant dense<[7, 6]> : tensor<2xi64>
// CHECK-DAG: [[PADS:%.+]] = onnx.Constant dense<[0, 1, 0, 0]> : tensor<4xi64>
// CHECK-DAG: [[POISON:%.+]] = ub.poison : tensor<f32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]], [[START]], [[END]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]], [[PADS]], [[POISON]],
// CHECK: return [[RESULT]]

// -----

func.func @negative_indices(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[-8]> : tensor<1xi64>
  %ends = onnx.Constant dense<[-2]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @negative_indices(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @clamped_end(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[3]> : tensor<1xi64>
  %ends = onnx.Constant dense<[9223372036854775807]> : tensor<1xi64>
  %pads = onnx.Constant dense<[3, 0]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<7xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<7xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @clamped_end(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @zero_fill(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[3]> : tensor<1xi64>
  %ends = onnx.Constant dense<[10]> : tensor<1xi64>
  %pads = onnx.Constant dense<[3, 0]> : tensor<2xi64>
  %fill = onnx.Constant dense<0.0> : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<7xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<7xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @zero_fill(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @reflect_mode(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[3]> : tensor<1xi64>
  %ends = onnx.Constant dense<[10]> : tensor<1xi64>
  %pads = onnx.Constant dense<[3, 0]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<7xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "reflect"} : (tensor<7xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @reflect_mode(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @strided_slice(%x: tensor<10xf32>) -> (tensor<7xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %steps = onnx.Constant dense<[2]> : tensor<1xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %steps) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, tensor<1xi64>) -> tensor<3xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<3xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<7xf32>
  return %result : tensor<7xf32>
}

// CHECK-LABEL: func.func @strided_slice(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @multiple_uses(%x: tensor<10xf32>) -> (tensor<10xf32>, tensor<6xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result, %slice : tensor<10xf32>, tensor<6xf32>
}

// CHECK-LABEL: func.func @multiple_uses(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @no_backfill(%x: tensor<10xf32>) -> (tensor<10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[0]> : tensor<1xi64>
  %ends = onnx.Constant dense<[7]> : tensor<1xi64>
  %pads = onnx.Constant dense<[3, 0]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<7xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<7xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @no_backfill(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @grows_other_axis(%x: tensor<1x1008x1x1xf32>) -> (tensor<16x1000x1x1xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[0, 0, 0, 0]> : tensor<4xi64>
  %ends = onnx.Constant dense<[1, 1000, 1, 1]> : tensor<4xi64>
  %pads = onnx.Constant dense<[0, 0, 0, 0, 15, 0, 0, 0]> : tensor<8xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<1x1008x1x1xf32>, tensor<4xi64>, tensor<4xi64>, none, none) -> tensor<1x1000x1x1xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<1x1000x1x1xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<16x1000x1x1xf32>
  return %result : tensor<16x1000x1x1xf32>
}

// CHECK-LABEL: func.func @grows_other_axis(
// CHECK-SAME: [[X:%.+]]: tensor<1x1008x1x1xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @negative_pad(%x: tensor<10xf32>) -> (tensor<6xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<6xf32>
  return %result : tensor<6xf32>
}

// CHECK-LABEL: func.func @negative_pad(
// CHECK-SAME: [[X:%.+]]: tensor<10xf32>
// CHECK: [[SLICE:%.+]] = "onnx.Slice"([[X]],
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

func.func @explicit_negative_axis(%x: tensor<2x10xf32>) -> (tensor<2x10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %axes, %none) : (tensor<2x10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, none) -> tensor<2x6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %axes) {mode = "constant"} : (tensor<2x6xf32>, tensor<2xi64>, tensor<f32>, tensor<1xi64>) -> tensor<2x10xf32>
  return %result : tensor<2x10xf32>
}

// CHECK-LABEL: func.func @explicit_negative_axis(
// CHECK-SAME: [[X:%.+]]: tensor<2x10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

func.func @pad_subset_axis(%x: tensor<2x10xf32>) -> (tensor<2x10xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 8]> : tensor<2xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %pad_axes = onnx.Constant dense<[1]> : tensor<1xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<2x10xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<2x6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %pad_axes) {mode = "constant"} : (tensor<2x6xf32>, tensor<2xi64>, tensor<f32>, tensor<1xi64>) -> tensor<2x10xf32>
  return %result : tensor<2x10xf32>
}

// CHECK-LABEL: func.func @pad_subset_axis(
// CHECK-SAME: [[X:%.+]]: tensor<2x10xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

// Constant pads alone are insufficient: the Slice indices must also be known.
func.func @dynamic_starts(%x: tensor<10xf32>, %starts: tensor<1xi64>) -> tensor<10xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, tensor<f32>, none) -> tensor<10xf32>
  return %result : tensor<10xf32>
}

// CHECK-LABEL: func.func @dynamic_starts(
// CHECK: [[SLICE:%.+]] = "onnx.Slice"
// CHECK: [[RESULT:%.+]] = "onnx.Pad"([[SLICE]],
// CHECK: return [[RESULT]]

// -----

// Integer data has the same poison-refinement semantics.
func.func @cancel_i8(%x: tensor<10xi8>) -> tensor<10xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 2]> : tensor<2xi64>
  %fill = ub.poison : tensor<i8>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10xi8>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6xi8>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6xi8>, tensor<2xi64>, tensor<i8>, none) -> tensor<10xi8>
  return %result : tensor<10xi8>
}

// CHECK-LABEL: func.func @cancel_i8(
// CHECK-SAME: [[X:%.+]]: tensor<10xi8>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]

// -----

// Omitted axes cover len(starts) leading dimensions, leaving the rest intact.
func.func @omitted_axes_subset(%x: tensor<10x2xf32>) -> tensor<10x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[2]> : tensor<1xi64>
  %ends = onnx.Constant dense<[8]> : tensor<1xi64>
  %pads = onnx.Constant dense<[2, 0, 2, 0]> : tensor<4xi64>
  %fill = ub.poison : tensor<f32>
  %slice = "onnx.Slice"(%x, %starts, %ends, %none, %none) : (tensor<10x2xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<6x2xf32>
  %result = "onnx.Pad"(%slice, %pads, %fill, %none) {mode = "constant"} : (tensor<6x2xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<10x2xf32>
  return %result : tensor<10x2xf32>
}

// CHECK-LABEL: func.func @omitted_axes_subset(
// CHECK-SAME: [[X:%.+]]: tensor<10x2xf32>
// CHECK-NOT: "onnx.Slice"
// CHECK-NOT: "onnx.Pad"
// CHECK: return [[X]]
