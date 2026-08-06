// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=OFF
// RUN: onnx-mlir-opt --decompose-onnx="enable-lstm-decomposition" %s -split-input-file | FileCheck %s --check-prefixes=ON,DIRECT,LOOP,REVERSE,BIDIR,ACT

// Direct single-step expansion. Missing B, initial states, and P exercise typed
// zero synthesis; omitted Y_h/Y_c exercise optional output preservation.
func.func @forward_seq1_optional_outputs(
    %x: tensor<1x2x3xf32>, %w: tensor<1x16x3xf32>,
    %r: tensor<1x16x4xf32>) -> (tensor<1x1x2x4xf32>, none, none) {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {direction = "forward", hidden_size = 4 : si64, layout = 0 : si64}
      : (tensor<1x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>,
         none, none, none, none, none)
      -> (tensor<1x1x2x4xf32>, none, none)
  return %y, %yh, %yc : tensor<1x1x2x4xf32>, none, none
}

// DIRECT-LABEL:  func.func @forward_seq1_optional_outputs
// DIRECT-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3xf32>, [[PARAM_1_:%.+]]: tensor<1x16x3xf32>, [[PARAM_2_:%.+]]: tensor<1x16x4xf32>) -> (tensor<1x1x2x4xf32>, none, none) {
// DIRECT-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// DIRECT-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 16]> : tensor<2xi64>
// DIRECT-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<[1, 2, 16]> : tensor<3xi64>
// DIRECT-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// DIRECT-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<32> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<16> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<12xf32>
// DIRECT-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<32xf32>
// DIRECT-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// DIRECT-DAG:       [[VAR_15_:%.+]] = "onnx.Squeeze"([[PARAM_1_]], [[VAR_14_]]) : (tensor<1x16x3xf32>, tensor<1xi64>) -> tensor<16x3xf32>
// DIRECT-DAG:       [[VAR_16_:%.+]] = "onnx.Squeeze"([[PARAM_2_]], [[VAR_14_]]) : (tensor<1x16x4xf32>, tensor<1xi64>) -> tensor<16x4xf32>
// DIRECT-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_15_]]) {perm = [1, 0]} : (tensor<16x3xf32>) -> tensor<3x16xf32>
// DIRECT-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_16_]]) {perm = [1, 0]} : (tensor<16x4xf32>) -> tensor<4x16xf32>
// DIRECT-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_14_]], [[VAR_11_]], [[VAR_14_]], [[VAR_10_]]) : (tensor<32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<16xf32>
// DIRECT-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_13_]], [[VAR_11_]], [[VAR_9_]], [[VAR_14_]], [[VAR_10_]]) : (tensor<32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<16xf32>
// DIRECT-DAG:       [[VAR_21_:%.+]] = "onnx.Add"([[VAR_19_]], [[VAR_20_]]) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
// DIRECT-DAG:       [[VAR_22_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_8_]]) {allowzero = 0 : si64} : (tensor<1x2x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
// DIRECT:           [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_17_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// DIRECT:           [[VAR_24_:%.+]] = "onnx.Add"([[VAR_23_]], [[VAR_21_]]) : (tensor<2x16xf32>, tensor<16xf32>) -> tensor<2x16xf32>
// DIRECT-DAG:       [[VAR_25_:%.+]] = "onnx.Reshape"([[VAR_24_]], [[VAR_7_]]) {allowzero = 0 : si64} : (tensor<2x16xf32>, tensor<3xi64>) -> tensor<1x2x16xf32>
// DIRECT-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_14_]], [[VAR_6_]], [[VAR_14_]], [[VAR_10_]]) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
// DIRECT-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_6_]], [[VAR_5_]], [[VAR_14_]], [[VAR_10_]]) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
// DIRECT-DAG:       [[VAR_28_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_5_]], [[VAR_4_]], [[VAR_14_]], [[VAR_10_]]) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
// DIRECT-DAG:       [[VAR_29_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x2x16xf32>, tensor<2xi64>) -> tensor<2x16xf32>
// DIRECT-DAG:       [[VAR_30_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_18_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// DIRECT:           [[VAR_31_:%.+]] = "onnx.Add"([[VAR_29_]], [[VAR_30_]]) : (tensor<2x16xf32>, tensor<2x16xf32>) -> tensor<2x16xf32>
// DIRECT-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_31_]], [[VAR_14_]], [[VAR_6_]], [[VAR_10_]], [[VAR_10_]]) : (tensor<2x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_33_:%.+]] = "onnx.Slice"([[VAR_31_]], [[VAR_6_]], [[VAR_5_]], [[VAR_10_]], [[VAR_10_]]) : (tensor<2x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_34_:%.+]] = "onnx.Slice"([[VAR_31_]], [[VAR_5_]], [[VAR_4_]], [[VAR_10_]], [[VAR_10_]]) : (tensor<2x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_35_:%.+]] = "onnx.Slice"([[VAR_31_]], [[VAR_4_]], [[VAR_11_]], [[VAR_10_]], [[VAR_10_]]) : (tensor<2x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_36_:%.+]] = "onnx.Mul"([[VAR_26_]], [[VAR_2_]]) : (tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_37_:%.+]] = "onnx.Mul"([[VAR_28_]], [[VAR_2_]]) : (tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_38_:%.+]] = "onnx.Add"([[VAR_32_]], [[VAR_36_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_39_:%.+]] = "onnx.Add"([[VAR_34_]], [[VAR_37_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_40_:%.+]] = "onnx.Sigmoid"([[VAR_38_]]) : (tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_41_:%.+]] = "onnx.Sigmoid"([[VAR_39_]]) : (tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_42_:%.+]] = "onnx.Tanh"([[VAR_35_]]) : (tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_43_:%.+]] = "onnx.Mul"([[VAR_41_]], [[VAR_2_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_44_:%.+]] = "onnx.Mul"([[VAR_40_]], [[VAR_42_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT:           [[VAR_45_:%.+]] = "onnx.Add"([[VAR_43_]], [[VAR_44_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT:           [[VAR_46_:%.+]] = "onnx.Mul"([[VAR_27_]], [[VAR_45_]]) : (tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT:           [[VAR_47_:%.+]] = "onnx.Add"([[VAR_33_]], [[VAR_46_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_48_:%.+]] = "onnx.Sigmoid"([[VAR_47_]]) : (tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT-DAG:       [[VAR_49_:%.+]] = "onnx.Tanh"([[VAR_45_]]) : (tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT:           [[VAR_50_:%.+]] = "onnx.Mul"([[VAR_48_]], [[VAR_49_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// DIRECT:           [[VAR_51_:%.+]] = "onnx.Unsqueeze"([[VAR_50_]], [[VAR_14_]]) : (tensor<2x4xf32>, tensor<1xi64>) -> tensor<1x2x4xf32>
// DIRECT:           [[VAR_52_:%.+]] = "onnx.Unsqueeze"([[VAR_51_]], [[VAR_10_]]) : (tensor<1x2x4xf32>, tensor<1xi64>) -> tensor<1x1x2x4xf32>
// DIRECT:           return [[VAR_52_]], [[VAR_0_]], [[VAR_0_]] : tensor<1x1x2x4xf32>, none, none
// DIRECT:         }
// OFF-LABEL: @forward_seq1_optional_outputs
// OFF: "onnx.LSTM"
// -----

// Multi-step expansion uses an onnx.Loop; sequence_lens causes state masks.
func.func @forward_seq4_masked(
    %x: tensor<4x3x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>, %lens: tensor<3xi32>)
    -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %lens, %none, %none, %none)
      {direction = "forward", hidden_size = 3 : si64, layout = 0 : si64}
      : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, tensor<3xi32>, none, none, none)
      -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  return %y, %yh, %yc : tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>
}

// LOOP-LABEL:  func.func @forward_seq4_masked
// LOOP-SAME:   ([[PARAM_0_:%.+]]: tensor<4x3x2xf32>, [[PARAM_1_:%.+]]: tensor<1x12x2xf32>, [[PARAM_2_:%.+]]: tensor<1x12x3xf32>, [[PARAM_3_:%.+]]: tensor<3xi32>) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>) {
// LOOP-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// LOOP-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// LOOP-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<4> : tensor<i64>
// LOOP-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0> : tensor<i32>
// LOOP-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<3x3xf32>
// LOOP-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<3x3xf32>
// LOOP-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<3x3xf32>
// LOOP-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<9> : tensor<1xi64>
// LOOP-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// LOOP-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// LOOP-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[4, 3, 12]> : tensor<3xi64>
// LOOP-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<[12, 2]> : tensor<2xi64>
// LOOP-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<24> : tensor<1xi64>
// LOOP-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// LOOP-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
// LOOP-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<9xf32>
// LOOP-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<24xf32>
// LOOP-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// LOOP-DAG:       [[VAR_18_:%.+]] = "onnx.Squeeze"([[PARAM_1_]], [[VAR_17_]]) : (tensor<1x12x2xf32>, tensor<1xi64>) -> tensor<12x2xf32>
// LOOP-DAG:       [[VAR_19_:%.+]] = "onnx.Squeeze"([[PARAM_2_]], [[VAR_17_]]) : (tensor<1x12x3xf32>, tensor<1xi64>) -> tensor<12x3xf32>
// LOOP-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_18_]]) {perm = [1, 0]} : (tensor<12x2xf32>) -> tensor<2x12xf32>
// LOOP-DAG:       [[VAR_21_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// LOOP-DAG:       [[VAR_22_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_17_]], [[VAR_14_]], [[VAR_17_]], [[VAR_13_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// LOOP-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_14_]], [[VAR_12_]], [[VAR_17_]], [[VAR_13_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// LOOP-DAG:       [[VAR_24_:%.+]] = "onnx.Add"([[VAR_22_]], [[VAR_23_]]) : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xf32>
// LOOP-DAG:       [[VAR_25_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_11_]]) {allowzero = 0 : si64} : (tensor<4x3x2xf32>, tensor<2xi64>) -> tensor<12x2xf32>
// LOOP:           [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_20_]]) : (tensor<12x2xf32>, tensor<2x12xf32>) -> tensor<12x12xf32>
// LOOP:           [[VAR_27_:%.+]] = "onnx.Add"([[VAR_26_]], [[VAR_24_]]) : (tensor<12x12xf32>, tensor<12xf32>) -> tensor<12x12xf32>
// LOOP-DAG:       [[VAR_28_:%.+]] = "onnx.Reshape"([[VAR_27_]], [[VAR_10_]]) {allowzero = 0 : si64} : (tensor<12x12xf32>, tensor<3xi64>) -> tensor<4x3x12xf32>
// LOOP-DAG:       [[VAR_29_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_17_]], [[VAR_9_]], [[VAR_17_]], [[VAR_13_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// LOOP-DAG:       [[VAR_30_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_9_]], [[VAR_8_]], [[VAR_17_]], [[VAR_13_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// LOOP-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_8_]], [[VAR_7_]], [[VAR_17_]], [[VAR_13_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// LOOP-DAG:       [[VAR_32_:%.+]] = "onnx.Less"([[VAR_3_]], [[PARAM_3_]]) : (tensor<i32>, tensor<3xi32>) -> tensor<3xi1>
// LOOP:           [[VAR_33_:%.+]] = "onnx.Unsqueeze"([[VAR_32_]], [[VAR_13_]]) : (tensor<3xi1>, tensor<1xi64>) -> tensor<3x1xi1>
// LOOP-DAG:       [[VAR_34_:%.+]] = "onnx.Where"([[VAR_33_]], [[VAR_6_]], [[VAR_4_]]) : (tensor<3x1xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:       [[VAR_35_:%.+]] = "onnx.Where"([[VAR_33_]], [[VAR_5_]], [[VAR_4_]]) : (tensor<3x1xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP:           [[VAR_36_:%.+]]:3 = "onnx.Loop"([[VAR_2_]], [[VAR_1_]], [[VAR_34_]], [[VAR_35_]]) ({
// LOOP:           ^bb0([[LOOP_ITER:%.+]]: tensor<i64>, [[LOOP_COND:%.+]]: tensor<i1>, [[LOOP_H:%.+]]: tensor<3x3xf32>, [[LOOP_C:%.+]]: tensor<3x3xf32>):
// LOOP-DAG:         [[VAR_40_:%.+]] = "onnx.Gather"([[VAR_28_]], [[LOOP_ITER]]) {axis = 0 : si64} : (tensor<4x3x12xf32>, tensor<i64>) -> tensor<3x12xf32>
// LOOP-DAG:         [[VAR_41_:%.+]] = "onnx.MatMul"([[LOOP_H]], [[VAR_21_]]) : (tensor<3x3xf32>, tensor<3x12xf32>) -> tensor<3x12xf32>
// LOOP:             [[VAR_42_:%.+]] = "onnx.Add"([[VAR_40_]], [[VAR_41_]]) : (tensor<3x12xf32>, tensor<3x12xf32>) -> tensor<3x12xf32>
// LOOP-DAG:         [[VAR_43_:%.+]] = "onnx.Slice"([[VAR_42_]], [[VAR_17_]], [[VAR_9_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<3x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_44_:%.+]] = "onnx.Slice"([[VAR_42_]], [[VAR_9_]], [[VAR_8_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<3x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_45_:%.+]] = "onnx.Slice"([[VAR_42_]], [[VAR_8_]], [[VAR_7_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<3x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_46_:%.+]] = "onnx.Slice"([[VAR_42_]], [[VAR_7_]], [[VAR_14_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<3x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_47_:%.+]] = "onnx.Mul"([[VAR_29_]], [[LOOP_C]]) : (tensor<3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_48_:%.+]] = "onnx.Mul"([[VAR_31_]], [[LOOP_C]]) : (tensor<3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_49_:%.+]] = "onnx.Add"([[VAR_43_]], [[VAR_47_]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_50_:%.+]] = "onnx.Add"([[VAR_45_]], [[VAR_48_]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_51_:%.+]] = "onnx.Sigmoid"([[VAR_49_]]) : (tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_52_:%.+]] = "onnx.Sigmoid"([[VAR_50_]]) : (tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_53_:%.+]] = "onnx.Tanh"([[VAR_46_]]) : (tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_54_:%.+]] = "onnx.Mul"([[VAR_52_]], [[LOOP_C]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_55_:%.+]] = "onnx.Mul"([[VAR_51_]], [[VAR_53_]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP:             [[VAR_56_:%.+]] = "onnx.Add"([[VAR_54_]], [[VAR_55_]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP:             [[VAR_57_:%.+]] = "onnx.Mul"([[VAR_30_]], [[VAR_56_]]) : (tensor<3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP:             [[VAR_58_:%.+]] = "onnx.Add"([[VAR_44_]], [[VAR_57_]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_59_:%.+]] = "onnx.Sigmoid"([[VAR_58_]]) : (tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_60_:%.+]] = "onnx.Tanh"([[VAR_56_]]) : (tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_61_:%.+]] = "onnx.Mul"([[VAR_59_]], [[VAR_60_]]) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_62_:%.+]] = "onnx.Cast"([[LOOP_ITER]]) {saturate = 1 : si64, to = i32} : (tensor<i64>) -> tensor<i32>
// LOOP:             [[VAR_63_:%.+]] = "onnx.Less"([[VAR_62_]], [[PARAM_3_]]) : (tensor<i32>, tensor<3xi32>) -> tensor<3xi1>
// LOOP:             [[VAR_64_:%.+]] = "onnx.Unsqueeze"([[VAR_63_]], [[VAR_13_]]) : (tensor<3xi1>, tensor<1xi64>) -> tensor<3x1xi1>
// LOOP-DAG:         [[VAR_65_:%.+]] = "onnx.Where"([[VAR_64_]], [[VAR_61_]], [[LOOP_H]]) : (tensor<3x1xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_66_:%.+]] = "onnx.Where"([[VAR_64_]], [[VAR_56_]], [[LOOP_C]]) : (tensor<3x1xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP-DAG:         [[VAR_67_:%.+]] = "onnx.Where"([[VAR_64_]], [[VAR_61_]], [[VAR_4_]]) : (tensor<3x1xi1>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
// LOOP:             onnx.Yield [[VAR_0_]], [[VAR_65_]], [[VAR_66_]], [[VAR_67_]] : tensor<i1>, tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>
// LOOP:           }) : (tensor<i64>, none, tensor<3x3xf32>, tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>, tensor<4x3x3xf32>)
// LOOP-DAG:       [[VAR_37_:%.+]] = "onnx.Unsqueeze"([[VAR_36_]]#2, [[VAR_13_]]) : (tensor<4x3x3xf32>, tensor<1xi64>) -> tensor<4x1x3x3xf32>
// LOOP-DAG:       [[VAR_38_:%.+]] = "onnx.Unsqueeze"([[VAR_36_]]#0, [[VAR_17_]]) : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<1x3x3xf32>
// LOOP-DAG:       [[VAR_39_:%.+]] = "onnx.Unsqueeze"([[VAR_36_]]#1, [[VAR_17_]]) : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<1x3x3xf32>
// LOOP:           return [[VAR_37_]], [[VAR_38_]], [[VAR_39_]] : tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>
// LOOP:         }
// -----

// Reverse, layout=1, clipping, coupled input/forget, and peepholes exercise
// layout normalization and the non-default recurrent attributes together.
func.func @reverse_layout1_attributes(
    %x: tensor<2x4x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>, %b: tensor<1x24xf32>,
    %h: tensor<2x1x3xf32>, %c: tensor<2x1x3xf32>, %p: tensor<1x9xf32>)
    -> (tensor<2x4x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %b, %none, %h, %c, %p)
      {clip = 2.0 : f32, direction = "reverse", hidden_size = 3 : si64,
       input_forget = 1 : si64, layout = 1 : si64}
      : (tensor<2x4x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         tensor<1x24xf32>, none, tensor<2x1x3xf32>, tensor<2x1x3xf32>,
         tensor<1x9xf32>)
      -> (tensor<2x4x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>)
  return %y, %yh, %yc : tensor<2x4x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>
}
// REVERSE-LABEL:  func.func @reverse_layout1_attributes
// REVERSE-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x2xf32>, [[PARAM_1_:%.+]]: tensor<1x12x2xf32>, [[PARAM_2_:%.+]]: tensor<1x12x3xf32>, [[PARAM_3_:%.+]]: tensor<1x24xf32>, [[PARAM_4_:%.+]]: tensor<2x1x3xf32>, [[PARAM_5_:%.+]]: tensor<2x1x3xf32>, [[PARAM_6_:%.+]]: tensor<1x9xf32>) -> (tensor<2x4x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>) {
// REVERSE-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// REVERSE-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// REVERSE-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// REVERSE-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<-2.000000e+00> : tensor<f32>
// REVERSE-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<9> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<3> : tensor<i64>
// REVERSE-DAG:       [[VAR_6_:%.+]] = "onnx.NoValue"() {value} : () -> none
// REVERSE-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<4> : tensor<i64>
// REVERSE-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<[4, 2, 12]> : tensor<3xi64>
// REVERSE-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<[8, 2]> : tensor<2xi64>
// REVERSE-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<24> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// REVERSE-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [1, 0, 2]} : (tensor<2x4x2xf32>) -> tensor<4x2x2xf32>
// REVERSE-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[PARAM_4_]]) {perm = [1, 0, 2]} : (tensor<2x1x3xf32>) -> tensor<1x2x3xf32>
// REVERSE-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[PARAM_5_]]) {perm = [1, 0, 2]} : (tensor<2x1x3xf32>) -> tensor<1x2x3xf32>
// REVERSE-DAG:       [[VAR_19_:%.+]] = "onnx.Squeeze"([[PARAM_1_]], [[VAR_15_]]) : (tensor<1x12x2xf32>, tensor<1xi64>) -> tensor<12x2xf32>
// REVERSE-DAG:       [[VAR_20_:%.+]] = "onnx.Squeeze"([[PARAM_2_]], [[VAR_15_]]) : (tensor<1x12x3xf32>, tensor<1xi64>) -> tensor<12x3xf32>
// REVERSE-DAG:       [[VAR_21_:%.+]] = "onnx.Squeeze"([[PARAM_3_]], [[VAR_15_]]) : (tensor<1x24xf32>, tensor<1xi64>) -> tensor<24xf32>
// REVERSE-DAG:       [[VAR_22_:%.+]] = "onnx.Squeeze"([[PARAM_6_]], [[VAR_15_]]) : (tensor<1x9xf32>, tensor<1xi64>) -> tensor<9xf32>
// REVERSE-DAG:       [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_19_]]) {perm = [1, 0]} : (tensor<12x2xf32>) -> tensor<2x12xf32>
// REVERSE-DAG:       [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_20_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// REVERSE-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_21_]], [[VAR_15_]], [[VAR_14_]], [[VAR_15_]], [[VAR_13_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// REVERSE-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_21_]], [[VAR_14_]], [[VAR_12_]], [[VAR_15_]], [[VAR_13_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// REVERSE-DAG:       [[VAR_27_:%.+]] = "onnx.Add"([[VAR_25_]], [[VAR_26_]]) : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xf32>
// REVERSE-DAG:       [[VAR_28_:%.+]] = "onnx.Reshape"([[VAR_16_]], [[VAR_11_]]) {allowzero = 0 : si64} : (tensor<4x2x2xf32>, tensor<2xi64>) -> tensor<8x2xf32>
// REVERSE:           [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_28_]], [[VAR_23_]]) : (tensor<8x2xf32>, tensor<2x12xf32>) -> tensor<8x12xf32>
// REVERSE:           [[VAR_30_:%.+]] = "onnx.Add"([[VAR_29_]], [[VAR_27_]]) : (tensor<8x12xf32>, tensor<12xf32>) -> tensor<8x12xf32>
// REVERSE-DAG:       [[VAR_31_:%.+]] = "onnx.Reshape"([[VAR_30_]], [[VAR_10_]]) {allowzero = 0 : si64} : (tensor<8x12xf32>, tensor<3xi64>) -> tensor<4x2x12xf32>
// REVERSE-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_22_]], [[VAR_15_]], [[VAR_9_]], [[VAR_15_]], [[VAR_13_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// REVERSE-DAG:       [[VAR_33_:%.+]] = "onnx.Slice"([[VAR_22_]], [[VAR_9_]], [[VAR_8_]], [[VAR_15_]], [[VAR_13_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// REVERSE-DAG:       [[VAR_34_:%.+]] = "onnx.Squeeze"([[VAR_17_]], [[VAR_15_]]) : (tensor<1x2x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
// REVERSE-DAG:       [[VAR_35_:%.+]] = "onnx.Squeeze"([[VAR_18_]], [[VAR_15_]]) : (tensor<1x2x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
// REVERSE:           [[VAR_36_:%.+]]:3 = "onnx.Loop"([[VAR_7_]], [[VAR_6_]], [[VAR_34_]], [[VAR_35_]]) ({
// REVERSE:           ^bb0([[REVERSE_ITER:%.+]]: tensor<i64>, [[REVERSE_COND:%.+]]: tensor<i1>, [[REVERSE_H:%.+]]: tensor<2x3xf32>, [[REVERSE_C:%.+]]: tensor<2x3xf32>):
// REVERSE:             [[VAR_43_:%.+]] = "onnx.Sub"([[VAR_5_]], [[REVERSE_ITER]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// REVERSE-DAG:         [[VAR_44_:%.+]] = "onnx.Gather"([[VAR_31_]], [[VAR_43_]]) {axis = 0 : si64} : (tensor<4x2x12xf32>, tensor<i64>) -> tensor<2x12xf32>
// REVERSE-DAG:         [[VAR_45_:%.+]] = "onnx.MatMul"([[REVERSE_H]], [[VAR_24_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// REVERSE:             [[VAR_46_:%.+]] = "onnx.Add"([[VAR_44_]], [[VAR_45_]]) : (tensor<2x12xf32>, tensor<2x12xf32>) -> tensor<2x12xf32>
// REVERSE-DAG:         [[VAR_47_:%.+]] = "onnx.Slice"([[VAR_46_]], [[VAR_15_]], [[VAR_9_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<2x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_48_:%.+]] = "onnx.Slice"([[VAR_46_]], [[VAR_9_]], [[VAR_8_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<2x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_49_:%.+]] = "onnx.Slice"([[VAR_46_]], [[VAR_4_]], [[VAR_14_]], [[VAR_13_]], [[VAR_13_]]) : (tensor<2x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_50_:%.+]] = "onnx.Mul"([[VAR_32_]], [[REVERSE_C]]) : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_51_:%.+]] = "onnx.Add"([[VAR_47_]], [[VAR_50_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_52_:%.+]] = "onnx.Clip"([[VAR_51_]], [[VAR_3_]], [[VAR_2_]]) : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_53_:%.+]] = "onnx.Sigmoid"([[VAR_52_]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_54_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_53_]]) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_55_:%.+]] = "onnx.Clip"([[VAR_49_]], [[VAR_3_]], [[VAR_2_]]) : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_56_:%.+]] = "onnx.Tanh"([[VAR_55_]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_57_:%.+]] = "onnx.Mul"([[VAR_54_]], [[REVERSE_C]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_58_:%.+]] = "onnx.Mul"([[VAR_53_]], [[VAR_56_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_59_:%.+]] = "onnx.Add"([[VAR_57_]], [[VAR_58_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_60_:%.+]] = "onnx.Mul"([[VAR_33_]], [[VAR_59_]]) : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_61_:%.+]] = "onnx.Add"([[VAR_48_]], [[VAR_60_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_62_:%.+]] = "onnx.Clip"([[VAR_61_]], [[VAR_3_]], [[VAR_2_]]) : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_63_:%.+]] = "onnx.Sigmoid"([[VAR_62_]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE-DAG:         [[VAR_64_:%.+]] = "onnx.Tanh"([[VAR_59_]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             [[VAR_65_:%.+]] = "onnx.Mul"([[VAR_63_]], [[VAR_64_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// REVERSE:             onnx.Yield [[VAR_0_]], [[VAR_65_]], [[VAR_59_]], [[VAR_65_]] : tensor<i1>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
// REVERSE:           }) : (tensor<i64>, none, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<4x2x3xf32>)
// REVERSE-DAG:       [[REVERSED_SCAN:%.+]] = "onnx.ReverseSequence"([[VAR_36_]]#2, [[REVERSED_LENS:%.+]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<4x2x3xf32>, tensor<2xi64>) -> tensor<4x2x3xf32>
// REVERSE-DAG:       [[VAR_37_:%.+]] = "onnx.Unsqueeze"([[REVERSED_SCAN]], [[VAR_13_]]) : (tensor<4x2x3xf32>, tensor<1xi64>) -> tensor<4x1x2x3xf32>
// REVERSE-DAG:       [[VAR_38_:%.+]] = "onnx.Unsqueeze"([[VAR_36_]]#0, [[VAR_15_]]) : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<1x2x3xf32>
// REVERSE-DAG:       [[VAR_39_:%.+]] = "onnx.Unsqueeze"([[VAR_36_]]#1, [[VAR_15_]]) : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<1x2x3xf32>
// REVERSE-DAG:       [[VAR_40_:%.+]] = "onnx.Transpose"([[VAR_37_]]) {perm = [2, 0, 1, 3]} : (tensor<4x1x2x3xf32>) -> tensor<2x4x1x3xf32>
// REVERSE-DAG:       [[VAR_41_:%.+]] = "onnx.Transpose"([[VAR_38_]]) {perm = [1, 0, 2]} : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
// REVERSE-DAG:       [[VAR_42_:%.+]] = "onnx.Transpose"([[VAR_39_]]) {perm = [1, 0, 2]} : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
// REVERSE:           return [[VAR_40_]], [[VAR_41_]], [[VAR_42_]] : tensor<2x4x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>
// REVERSE:         }
// -----

// Two Loop bodies exercise the ONNX activation matrix. ONNX orders
// activation arrays as forward f/g/h followed by reverse f/g/h.
func.func @bidirectional_activations(%x: tensor<2x1x2xf32>,
    %w: tensor<2x12x2xf32>, %r: tensor<2x12x3xf32>)
    -> (tensor<2x2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activation_alpha = [0.25 : f32, 1.5 : f32, 2.0 : f32, 0.125 : f32,
                           0.75 : f32, 1.25 : f32],
       activation_beta = [0.75 : f32, 0.5 : f32, 0.25 : f32, 1.0 : f32,
                          1.0 : f32, 1.0 : f32],
       activations = ["HardSigmoid", "Affine", "ScaledTanh", "LeakyRelu",
                      "ThresholdedRelu", "Elu"],
       direction = "bidirectional", hidden_size = 3 : si64, layout = 0 : si64}
      : (tensor<2x1x2xf32>, tensor<2x12x2xf32>, tensor<2x12x3xf32>,
         none, none, none, none, none)
      -> (tensor<2x2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>)
  return %y, %yh, %yc : tensor<2x2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>
}
// BIDIR-LABEL:  func.func @bidirectional_activations
// BIDIR-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x2xf32>, [[PARAM_1_:%.+]]: tensor<2x12x2xf32>, [[PARAM_2_:%.+]]: tensor<2x12x3xf32>) -> (tensor<2x2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>) {
// BIDIR-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i64>
// BIDIR-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// BIDIR-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// BIDIR-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2.500000e-01> : tensor<f32>
// BIDIR-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// BIDIR-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<1.500000e+00> : tensor<f32>
// BIDIR-DAG:       [[VAR_7_:%.+]] = "onnx.NoValue"() {value} : () -> none
// BIDIR-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<2> : tensor<i64>
// BIDIR-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<9> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<[2, 1, 12]> : tensor<3xi64>
// BIDIR-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// BIDIR-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<24> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<12> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<2x1x3xf32>
// BIDIR-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<2x24xf32>
// BIDIR-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<2x9xf32>
// BIDIR-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// BIDIR-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// BIDIR:           [[VAR_21_:%.+]] = "onnx.Slice"([[PARAM_1_]], [[VAR_19_]], [[VAR_20_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x12x2xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x12x2xf32>
// BIDIR-DAG:       [[VAR_22_:%.+]] = "onnx.Squeeze"([[VAR_21_]], [[VAR_19_]]) : (tensor<1x12x2xf32>, tensor<1xi64>) -> tensor<12x2xf32>
// BIDIR-DAG:       [[VAR_23_:%.+]] = "onnx.Slice"([[PARAM_2_]], [[VAR_19_]], [[VAR_20_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x12x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x12x3xf32>
// BIDIR-DAG:       [[VAR_24_:%.+]] = "onnx.Squeeze"([[VAR_23_]], [[VAR_19_]]) : (tensor<1x12x3xf32>, tensor<1xi64>) -> tensor<12x3xf32>
// BIDIR-DAG:       [[VAR_25_:%.+]] = "onnx.Slice"([[VAR_17_]], [[VAR_19_]], [[VAR_20_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24xf32>
// BIDIR-DAG:       [[VAR_26_:%.+]] = "onnx.Squeeze"([[VAR_25_]], [[VAR_19_]]) : (tensor<1x24xf32>, tensor<1xi64>) -> tensor<24xf32>
// BIDIR-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_18_]], [[VAR_19_]], [[VAR_20_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x9xf32>
// BIDIR-DAG:       [[VAR_28_:%.+]] = "onnx.Squeeze"([[VAR_27_]], [[VAR_19_]]) : (tensor<1x9xf32>, tensor<1xi64>) -> tensor<9xf32>
// BIDIR-DAG:       [[VAR_29_:%.+]] = "onnx.Transpose"([[VAR_22_]]) {perm = [1, 0]} : (tensor<12x2xf32>) -> tensor<2x12xf32>
// BIDIR-DAG:       [[VAR_30_:%.+]] = "onnx.Transpose"([[VAR_24_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// BIDIR-DAG:       [[VAR_31_:%.+]] = "onnx.Slice"([[VAR_26_]], [[VAR_19_]], [[VAR_15_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// BIDIR-DAG:       [[VAR_32_:%.+]] = "onnx.Slice"([[VAR_26_]], [[VAR_15_]], [[VAR_14_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// BIDIR-DAG:       [[VAR_33_:%.+]] = "onnx.Add"([[VAR_31_]], [[VAR_32_]]) : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xf32>
// BIDIR-DAG:       [[VAR_34_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_13_]]) {allowzero = 0 : si64} : (tensor<2x1x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>
// BIDIR:           [[VAR_35_:%.+]] = "onnx.MatMul"([[VAR_34_]], [[VAR_29_]]) : (tensor<2x2xf32>, tensor<2x12xf32>) -> tensor<2x12xf32>
// BIDIR:           [[VAR_36_:%.+]] = "onnx.Add"([[VAR_35_]], [[VAR_33_]]) : (tensor<2x12xf32>, tensor<12xf32>) -> tensor<2x12xf32>
// BIDIR-DAG:       [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_36_]], [[VAR_12_]]) {allowzero = 0 : si64} : (tensor<2x12xf32>, tensor<3xi64>) -> tensor<2x1x12xf32>
// BIDIR-DAG:       [[VAR_38_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_19_]], [[VAR_11_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// BIDIR-DAG:       [[VAR_39_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_11_]], [[VAR_10_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// BIDIR-DAG:       [[VAR_40_:%.+]] = "onnx.Slice"([[VAR_28_]], [[VAR_10_]], [[VAR_9_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// BIDIR-DAG:       [[VAR_41_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_19_]], [[VAR_20_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x1x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR-DAG:       [[VAR_42_:%.+]] = "onnx.Squeeze"([[VAR_41_]], [[VAR_19_]]) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:       [[VAR_43_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_19_]], [[VAR_20_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x1x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR:           [[VAR_44_:%.+]] = "onnx.Squeeze"([[VAR_43_]], [[VAR_19_]]) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR:           [[VAR_45_:%.+]]:3 = "onnx.Loop"([[VAR_8_]], [[VAR_7_]], [[VAR_42_]], [[VAR_44_]]) ({
// BIDIR:           ^bb0([[F_ITER:%.+]]: tensor<i64>, [[F_COND:%.+]]: tensor<i1>, [[F_H:%.+]]: tensor<1x3xf32>, [[F_C:%.+]]: tensor<1x3xf32>):
// BIDIR-DAG:         [[VAR_80_:%.+]] = "onnx.Gather"([[VAR_37_]], [[F_ITER]]) {axis = 0 : si64} : (tensor<2x1x12xf32>, tensor<i64>) -> tensor<1x12xf32>
// BIDIR-DAG:         [[VAR_81_:%.+]] = "onnx.MatMul"([[F_H]], [[VAR_30_]]) : (tensor<1x3xf32>, tensor<3x12xf32>) -> tensor<1x12xf32>
// BIDIR:             [[VAR_82_:%.+]] = "onnx.Add"([[VAR_80_]], [[VAR_81_]]) : (tensor<1x12xf32>, tensor<1x12xf32>) -> tensor<1x12xf32>
// BIDIR-DAG:         [[VAR_83_:%.+]] = "onnx.Slice"([[VAR_82_]], [[VAR_19_]], [[VAR_11_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_84_:%.+]] = "onnx.Slice"([[VAR_82_]], [[VAR_11_]], [[VAR_10_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_85_:%.+]] = "onnx.Slice"([[VAR_82_]], [[VAR_10_]], [[VAR_9_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_86_:%.+]] = "onnx.Slice"([[VAR_82_]], [[VAR_9_]], [[VAR_15_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_87_:%.+]] = "onnx.Mul"([[VAR_38_]], [[F_C]]) : (tensor<3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_88_:%.+]] = "onnx.Mul"([[VAR_40_]], [[F_C]]) : (tensor<3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_89_:%.+]] = "onnx.Add"([[VAR_83_]], [[VAR_87_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_90_:%.+]] = "onnx.Add"([[VAR_85_]], [[VAR_88_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_91_:%.+]] = "onnx.HardSigmoid"([[VAR_89_]]) {alpha = 2.500000e-01 : f32, beta = 7.500000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_92_:%.+]] = "onnx.HardSigmoid"([[VAR_90_]]) {alpha = 2.500000e-01 : f32, beta = 7.500000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_93_:%.+]] = "onnx.Mul"([[VAR_86_]], [[VAR_6_]]) : (tensor<1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_94_:%.+]] = "onnx.Add"([[VAR_93_]], [[VAR_5_]]) : (tensor<1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_95_:%.+]] = "onnx.Mul"([[VAR_92_]], [[F_C]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_96_:%.+]] = "onnx.Mul"([[VAR_91_]], [[VAR_94_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_97_:%.+]] = "onnx.Add"([[VAR_95_]], [[VAR_96_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_98_:%.+]] = "onnx.Mul"([[VAR_39_]], [[VAR_97_]]) : (tensor<3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_99_:%.+]] = "onnx.Add"([[VAR_84_]], [[VAR_98_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_100_:%.+]] = "onnx.HardSigmoid"([[VAR_99_]]) {alpha = 2.500000e-01 : f32, beta = 7.500000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_101_:%.+]] = "onnx.Mul"([[VAR_97_]], [[VAR_4_]]) : (tensor<1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_102_:%.+]] = "onnx.Tanh"([[VAR_101_]]) : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_103_:%.+]] = "onnx.Mul"([[VAR_102_]], [[VAR_3_]]) : (tensor<1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_104_:%.+]] = "onnx.Mul"([[VAR_100_]], [[VAR_103_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             onnx.Yield [[VAR_2_]], [[VAR_104_]], [[VAR_97_]], [[VAR_104_]] : tensor<i1>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>
// BIDIR:           }) : (tensor<i64>, none, tensor<1x3xf32>, tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<2x1x3xf32>)
// BIDIR:           [[VAR_46_:%.+]] = "onnx.Slice"([[PARAM_1_]], [[VAR_20_]], [[VAR_1_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x12x2xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x12x2xf32>
// BIDIR-DAG:       [[VAR_47_:%.+]] = "onnx.Squeeze"([[VAR_46_]], [[VAR_19_]]) : (tensor<1x12x2xf32>, tensor<1xi64>) -> tensor<12x2xf32>
// BIDIR-DAG:       [[VAR_48_:%.+]] = "onnx.Slice"([[PARAM_2_]], [[VAR_20_]], [[VAR_1_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x12x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x12x3xf32>
// BIDIR-DAG:       [[VAR_49_:%.+]] = "onnx.Squeeze"([[VAR_48_]], [[VAR_19_]]) : (tensor<1x12x3xf32>, tensor<1xi64>) -> tensor<12x3xf32>
// BIDIR-DAG:       [[VAR_50_:%.+]] = "onnx.Slice"([[VAR_17_]], [[VAR_20_]], [[VAR_1_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x24xf32>
// BIDIR-DAG:       [[VAR_51_:%.+]] = "onnx.Squeeze"([[VAR_50_]], [[VAR_19_]]) : (tensor<1x24xf32>, tensor<1xi64>) -> tensor<24xf32>
// BIDIR-DAG:       [[VAR_52_:%.+]] = "onnx.Slice"([[VAR_18_]], [[VAR_20_]], [[VAR_1_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x9xf32>
// BIDIR-DAG:       [[VAR_53_:%.+]] = "onnx.Squeeze"([[VAR_52_]], [[VAR_19_]]) : (tensor<1x9xf32>, tensor<1xi64>) -> tensor<9xf32>
// BIDIR-DAG:       [[VAR_54_:%.+]] = "onnx.Transpose"([[VAR_47_]]) {perm = [1, 0]} : (tensor<12x2xf32>) -> tensor<2x12xf32>
// BIDIR-DAG:       [[VAR_55_:%.+]] = "onnx.Transpose"([[VAR_49_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// BIDIR-DAG:       [[VAR_56_:%.+]] = "onnx.Slice"([[VAR_51_]], [[VAR_19_]], [[VAR_15_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// BIDIR-DAG:       [[VAR_57_:%.+]] = "onnx.Slice"([[VAR_51_]], [[VAR_15_]], [[VAR_14_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<24xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<12xf32>
// BIDIR-DAG:       [[VAR_58_:%.+]] = "onnx.Add"([[VAR_56_]], [[VAR_57_]]) : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xf32>
// BIDIR-DAG:       [[VAR_59_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_13_]]) {allowzero = 0 : si64} : (tensor<2x1x2xf32>, tensor<2xi64>) -> tensor<2x2xf32>
// BIDIR:           [[VAR_60_:%.+]] = "onnx.MatMul"([[VAR_59_]], [[VAR_54_]]) : (tensor<2x2xf32>, tensor<2x12xf32>) -> tensor<2x12xf32>
// BIDIR:           [[VAR_61_:%.+]] = "onnx.Add"([[VAR_60_]], [[VAR_58_]]) : (tensor<2x12xf32>, tensor<12xf32>) -> tensor<2x12xf32>
// BIDIR-DAG:       [[VAR_62_:%.+]] = "onnx.Reshape"([[VAR_61_]], [[VAR_12_]]) {allowzero = 0 : si64} : (tensor<2x12xf32>, tensor<3xi64>) -> tensor<2x1x12xf32>
// BIDIR-DAG:       [[VAR_63_:%.+]] = "onnx.Slice"([[VAR_53_]], [[VAR_19_]], [[VAR_11_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// BIDIR-DAG:       [[VAR_64_:%.+]] = "onnx.Slice"([[VAR_53_]], [[VAR_11_]], [[VAR_10_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// BIDIR-DAG:       [[VAR_65_:%.+]] = "onnx.Slice"([[VAR_53_]], [[VAR_10_]], [[VAR_9_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// BIDIR-DAG:       [[VAR_66_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_20_]], [[VAR_1_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x1x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR-DAG:       [[VAR_67_:%.+]] = "onnx.Squeeze"([[VAR_66_]], [[VAR_19_]]) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:       [[VAR_68_:%.+]] = "onnx.Slice"([[VAR_16_]], [[VAR_20_]], [[VAR_1_]], [[VAR_19_]], [[VAR_20_]]) : (tensor<2x1x3xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR:           [[VAR_69_:%.+]] = "onnx.Squeeze"([[VAR_68_]], [[VAR_19_]]) : (tensor<1x1x3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR:           [[VAR_70_:%.+]]:3 = "onnx.Loop"([[VAR_8_]], [[VAR_7_]], [[VAR_67_]], [[VAR_69_]]) ({
// BIDIR:           ^bb0([[R_ITER:%.+]]: tensor<i64>, [[R_COND:%.+]]: tensor<i1>, [[R_H:%.+]]: tensor<1x3xf32>, [[R_C:%.+]]: tensor<1x3xf32>):
// BIDIR:             [[VAR_80_1_:%.+]] = "onnx.Sub"([[VAR_0_]], [[R_ITER]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// BIDIR-DAG:         [[VAR_81_1_:%.+]] = "onnx.Gather"([[VAR_62_]], [[VAR_80_1_]]) {axis = 0 : si64} : (tensor<2x1x12xf32>, tensor<i64>) -> tensor<1x12xf32>
// BIDIR-DAG:         [[VAR_82_1_:%.+]] = "onnx.MatMul"([[R_H]], [[VAR_55_]]) : (tensor<1x3xf32>, tensor<3x12xf32>) -> tensor<1x12xf32>
// BIDIR:             [[VAR_83_1_:%.+]] = "onnx.Add"([[VAR_81_1_]], [[VAR_82_1_]]) : (tensor<1x12xf32>, tensor<1x12xf32>) -> tensor<1x12xf32>
// BIDIR-DAG:         [[VAR_84_1_:%.+]] = "onnx.Slice"([[VAR_83_1_]], [[VAR_19_]], [[VAR_11_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_85_1_:%.+]] = "onnx.Slice"([[VAR_83_1_]], [[VAR_11_]], [[VAR_10_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_86_1_:%.+]] = "onnx.Slice"([[VAR_83_1_]], [[VAR_10_]], [[VAR_9_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_87_1_:%.+]] = "onnx.Slice"([[VAR_83_1_]], [[VAR_9_]], [[VAR_15_]], [[VAR_20_]], [[VAR_20_]]) : (tensor<1x12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_88_1_:%.+]] = "onnx.Mul"([[VAR_63_]], [[R_C]]) : (tensor<3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_89_1_:%.+]] = "onnx.Mul"([[VAR_65_]], [[R_C]]) : (tensor<3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_90_1_:%.+]] = "onnx.Add"([[VAR_84_1_]], [[VAR_88_1_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_91_1_:%.+]] = "onnx.Add"([[VAR_86_1_]], [[VAR_89_1_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_92_1_:%.+]] = "onnx.LeakyRelu"([[VAR_90_1_]]) {alpha = 1.250000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_93_1_:%.+]] = "onnx.LeakyRelu"([[VAR_91_1_]]) {alpha = 1.250000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_94_1_:%.+]] = "onnx.ThresholdedRelu"([[VAR_87_1_]]) {alpha = 7.500000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_95_1_:%.+]] = "onnx.Mul"([[VAR_93_1_]], [[R_C]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_96_1_:%.+]] = "onnx.Mul"([[VAR_92_1_]], [[VAR_94_1_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_97_1_:%.+]] = "onnx.Add"([[VAR_95_1_]], [[VAR_96_1_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_98_1_:%.+]] = "onnx.Mul"([[VAR_64_]], [[VAR_97_1_]]) : (tensor<3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_99_1_:%.+]] = "onnx.Add"([[VAR_85_1_]], [[VAR_98_1_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_100_1_:%.+]] = "onnx.LeakyRelu"([[VAR_99_1_]]) {alpha = 1.250000e-01 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR-DAG:         [[VAR_101_1_:%.+]] = "onnx.Elu"([[VAR_97_1_]]) {alpha = 1.250000e+00 : f32} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             [[VAR_102_1_:%.+]] = "onnx.Mul"([[VAR_100_1_]], [[VAR_101_1_]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// BIDIR:             onnx.Yield [[VAR_2_]], [[VAR_102_1_]], [[VAR_97_1_]], [[VAR_102_1_]] : tensor<i1>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>
// BIDIR:           }) : (tensor<i64>, none, tensor<1x3xf32>, tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<2x1x3xf32>)
// BIDIR-DAG:       [[VAR_71_:%.+]] = "onnx.Unsqueeze"([[VAR_45_]]#2, [[VAR_20_]]) : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x1x1x3xf32>
// BIDIR-DAG:       [[VAR_72_:%.+]] = "onnx.Unsqueeze"([[VAR_45_]]#0, [[VAR_19_]]) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR-DAG:       [[VAR_73_:%.+]] = "onnx.Unsqueeze"([[VAR_45_]]#1, [[VAR_19_]]) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR-DAG:       [[REVERSED_SCAN:%.+]] = "onnx.ReverseSequence"([[VAR_70_]]#2, [[REVERSED_LENS:%.+]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
// BIDIR-DAG:       [[VAR_74_:%.+]] = "onnx.Unsqueeze"([[REVERSED_SCAN]], [[VAR_20_]]) : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x1x1x3xf32>
// BIDIR-DAG:       [[VAR_75_:%.+]] = "onnx.Unsqueeze"([[VAR_70_]]#0, [[VAR_19_]]) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR-DAG:       [[VAR_76_:%.+]] = "onnx.Unsqueeze"([[VAR_70_]]#1, [[VAR_19_]]) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1x3xf32>
// BIDIR-DAG:       [[VAR_77_:%.+]] = "onnx.Concat"([[VAR_71_]], [[VAR_74_]]) {axis = 1 : si64} : (tensor<2x1x1x3xf32>, tensor<2x1x1x3xf32>) -> tensor<2x2x1x3xf32>
// BIDIR-DAG:       [[VAR_78_:%.+]] = "onnx.Concat"([[VAR_72_]], [[VAR_75_]]) {axis = 0 : si64} : (tensor<1x1x3xf32>, tensor<1x1x3xf32>) -> tensor<2x1x3xf32>
// BIDIR-DAG:       [[VAR_79_:%.+]] = "onnx.Concat"([[VAR_73_]], [[VAR_76_]]) {axis = 0 : si64} : (tensor<1x1x3xf32>, tensor<1x1x3xf32>) -> tensor<2x1x3xf32>
// BIDIR:           return [[VAR_77_]], [[VAR_78_]], [[VAR_79_]] : tensor<2x2x1x3xf32>, tensor<2x1x3xf32>, tensor<2x1x3xf32>
// BIDIR:         }
// -----

func.func @activation_relu(%x: tensor<1x1x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activations = ["Relu"], direction = "forward", hidden_size = 3 : si64,
       layout = 0 : si64}
      : (tensor<1x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x1x3xf32>, none, none)
  return %y : tensor<1x1x1x3xf32>
}
// ACT-LABEL: @activation_relu
// ACT-NOT: "onnx.LSTM"
// ACT: "onnx.Relu"

// -----

func.func @activation_tanh(%x: tensor<1x1x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activations = ["Tanh"], direction = "forward", hidden_size = 3 : si64,
       layout = 0 : si64}
      : (tensor<1x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x1x3xf32>, none, none)
  return %y : tensor<1x1x1x3xf32>
}
// ACT-LABEL: @activation_tanh
// ACT-NOT: "onnx.LSTM"
// ACT: "onnx.Tanh"

// -----

func.func @activation_sigmoid(%x: tensor<1x1x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activations = ["Sigmoid"], direction = "forward", hidden_size = 3 : si64,
       layout = 0 : si64}
      : (tensor<1x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x1x3xf32>, none, none)
  return %y : tensor<1x1x1x3xf32>
}
// ACT-LABEL: @activation_sigmoid
// ACT-NOT: "onnx.LSTM"
// ACT: "onnx.Sigmoid"

// -----

func.func @activation_selu(%x: tensor<1x1x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activation_alpha = [1.25 : f32], activation_beta = [0.75 : f32],
       activations = ["Selu"], direction = "forward", hidden_size = 3 : si64,
       layout = 0 : si64}
      : (tensor<1x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x1x3xf32>, none, none)
  return %y : tensor<1x1x1x3xf32>
}
// ACT-LABEL: @activation_selu
// ACT-NOT: "onnx.LSTM"
// ACT: "onnx.Selu"{{.*}}alpha = 1.250000e+00{{.*}}gamma = 7.500000e-01

// -----

func.func @activation_softsign(%x: tensor<1x1x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activations = ["Softsign"], direction = "forward", hidden_size = 3 : si64,
       layout = 0 : si64}
      : (tensor<1x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x1x3xf32>, none, none)
  return %y : tensor<1x1x1x3xf32>
}
// ACT-LABEL: @activation_softsign
// ACT-NOT: "onnx.LSTM"
// ACT: "onnx.Softsign"

// -----

func.func @activation_softplus(%x: tensor<1x1x2xf32>, %w: tensor<1x12x2xf32>,
    %r: tensor<1x12x3xf32>) -> tensor<1x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {activations = ["Softplus"], direction = "forward", hidden_size = 3 : si64,
       layout = 0 : si64}
      : (tensor<1x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none) -> (tensor<1x1x1x3xf32>, none, none)
  return %y : tensor<1x1x1x3xf32>
}
// ACT-LABEL: @activation_softplus
// ACT-NOT: "onnx.LSTM"
// ACT: "onnx.Softplus"

// -----

func.func @dynamic_unchanged(%x: tensor<?x1x2xf32>,
    %w: tensor<1x12x2xf32>, %r: tensor<1x12x3xf32>)
    -> tensor<?x1x1x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %yh, %yc = "onnx.LSTM"(%x, %w, %r, %none, %none, %none, %none, %none)
      {direction = "forward", hidden_size = 3 : si64, layout = 0 : si64}
      : (tensor<?x1x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>,
         none, none, none, none, none)
      -> (tensor<?x1x1x3xf32>, none, none)
  return %y : tensor<?x1x1x3xf32>
}
// ON-LABEL: @dynamic_unchanged
// ON: "onnx.LSTM"
