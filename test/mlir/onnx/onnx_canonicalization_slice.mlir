// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s
//
// Tests for NormalizeSliceOperandsPattern: the canonical owner for static
// ONNX Slice operand normalization. Canonicalization materializes omitted
// axes/steps, expands partial-rank operands to full rank, and normalizes
// starts/ends/axes per ONNXSliceOpShapeHelper semantics.

// -----
// INT64_MAX end clamped to dim (64); explicit step=1 and axis=3 preserved.
func.func @slice_clamps_int64_max_end(%arg0: tensor<1x1600x3x64xf32>) -> tensor<1x1600x3x32xf32> {
  %starts = onnx.Constant dense<32> : tensor<1xi64>
  %ends   = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %axes   = onnx.Constant dense<3> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x1600x3x64xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1600x3x32xf32>
  return %0 : tensor<1x1600x3x32xf32>
// CHECK-LABEL:  func.func @slice_clamps_int64_max_end
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1600x3x64xf32>) -> tensor<1x1600x3x32xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[0, 0, 0, 32]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[1, 1600, 3, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<1> : tensor<4xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<1x1600x3x64xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x1600x3x32xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<1x1600x3x32xf32>
}

// -----
// Negative axis (-1 -> 2) and negative start (-2 -> 2) normalized to full rank.
func.func @slice_normalizes_negative_axis_and_start(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
  %starts = onnx.Constant dense<-2> : tensor<1xi64>
  %ends   = onnx.Constant dense<4> : tensor<1xi64>
  %axes   = onnx.Constant dense<-1> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3x2xf32>
  return %0 : tensor<2x3x2xf32>
// CHECK-LABEL:  func.func @slice_normalizes_negative_axis_and_start
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[0, 0, 2]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<2x3x4xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x3x2xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<2x3x2xf32>
}

// -----
// None axes materialized to [0]; None steps materialized to [1]; then expanded
// to full rank.
func.func @slice_materializes_none_axes_and_steps(%arg0: tensor<8xf32>) -> tensor<3xf32> {
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends   = onnx.Constant dense<5> : tensor<1xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %none, %none) : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>, none, none) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @slice_materializes_none_axes_and_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<8xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<3xf32>
}

// -----
// Already full-rank canonical: no rewrites expected.
func.func @slice_already_canonical(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
  %starts = onnx.Constant dense<[0, 0, 2]> : tensor<3xi64>
  %ends   = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
  %axes   = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %steps  = onnx.Constant dense<1> : tensor<3xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x3x4xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x3x2xf32>
  return %0 : tensor<2x3x2xf32>
// CHECK-LABEL: func.func @slice_already_canonical
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[0, 0, 2]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<2x3x4xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x3x2xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<2x3x2xf32>
}

// -----
// Overshoot end (100 > dim=6) clamped to 6 and expanded to full rank.
func.func @slice_clamps_overshoot_end(%arg0: tensor<5x6xf32>) -> tensor<5x4xf32> {
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends   = onnx.Constant dense<100> : tensor<1xi64>
  %axes   = onnx.Constant dense<1> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<5x6xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
// CHECK-LABEL:  func.func @slice_clamps_overshoot_end
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x6xf32>) -> tensor<5x4xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[5, 6]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<1> : tensor<2xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<5x6xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x4xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<5x4xf32>
}

// -----
// Dynamic input: normalization cannot run; operands remain unchanged.
func.func @slice_dynamic_data_unchanged(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %starts = onnx.Constant dense<-1> : tensor<1xi64>
  %ends   = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %axes   = onnx.Constant dense<1> : tensor<1xi64>
  %steps  = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @slice_dynamic_data_unchanged
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_2_]]) : (tensor<?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?xf32>
}

// -----
// Negative step: start normalized; end sentinel preserved; expanded to full rank.
func.func @slice_negative_step_axis_normalized_only(%arg0: tensor<3x2xi64>) -> tensor<3x2xi64> {
  %starts = onnx.Constant dense<-1> : tensor<1xi64>
  %ends   = onnx.Constant dense<-9223372036854775807> : tensor<1xi64>
  %axes   = onnx.Constant dense<0> : tensor<1xi64>
  %steps  = onnx.Constant dense<-1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) {onnx_node_name = "/Slice"} : (tensor<3x2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x2xi64>
  return %0 : tensor<3x2xi64>
// CHECK-LABEL:  func.func @slice_negative_step_axis_normalized_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xi64>) -> tensor<3x2xi64> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[2, 0]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[-4, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) {onnx_node_name = "/Slice"} : (tensor<3x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x2xi64>
// CHECK:           return [[VAR_SLICE_]] : tensor<3x2xi64>
}

// -----
func.func @slice_neg_step_start_at_dim_clamps_to_dim_minus_one(%arg0: tensor<4xf32>) -> tensor<3xf32> {
  %starts = onnx.Constant dense<4> : tensor<1xi64>
  %ends   = onnx.Constant dense<0> : tensor<1xi64>
  %axes   = onnx.Constant dense<0> : tensor<1xi64>
  %steps  = onnx.Constant dense<-1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @slice_neg_step_start_at_dim_clamps_to_dim_minus_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_ENDS_AXES_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_AXES_]], [[VAR_ENDS_AXES_]], [[VAR_STEPS_]]) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<3xf32>
}

// -----
// None axes/steps with partial-rank starts/ends: materialize then expand.
func.func @slice_none_axes_steps_n_less_than_rank(%arg0: tensor<2x4x6x8xf32>) -> tensor<1x3x6x8xf32> {
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends   = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %none, %none) : (tensor<2x4x6x8xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<1x3x6x8xf32>
  return %0 : tensor<1x3x6x8xf32>
// CHECK-LABEL:  func.func @slice_none_axes_steps_n_less_than_rank
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x6x8xf32>) -> tensor<1x3x6x8xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[1, 1, 0, 0]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[2, 4, 6, 8]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<1> : tensor<4xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<2x4x6x8xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x3x6x8xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<1x3x6x8xf32>
}

// -----
// Non-XMC regression equivalent to legacy standardize-slice-ops axes=[1,3] case.
func.func @slice_reordered_axes_expanded_to_full_rank(%arg0: tensor<10x20x30x40xf32>) -> tensor<10x5x30x9xf32> {
  %starts = onnx.Constant dense<[5, 10]> : tensor<2xi64>
  %ends = onnx.Constant dense<[15, 35]> : tensor<2xi64>
  %axes = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %result = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<10x20x30x40xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<10x5x30x9xf32>
  return %result : tensor<10x5x30x9xf32>
// CHECK-LABEL:  func.func @slice_reordered_axes_expanded_to_full_rank
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x20x30x40xf32>) -> tensor<10x5x30x9xf32> {
// CHECK-DAG:       [[VAR_STARTS_:%.+]] = onnx.Constant dense<[0, 5, 0, 10]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_ENDS_:%.+]] = onnx.Constant dense<[10, 15, 30, 35]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_AXES_:%.+]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_STEPS_:%.+]] = onnx.Constant dense<[1, 2, 1, 3]> : tensor<4xi64>
// CHECK:           [[VAR_SLICE_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_STARTS_]], [[VAR_ENDS_]], [[VAR_AXES_]], [[VAR_STEPS_]]) : (tensor<10x20x30x40xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<10x5x30x9xf32>
// CHECK:           return [[VAR_SLICE_]] : tensor<10x5x30x9xf32>
}
