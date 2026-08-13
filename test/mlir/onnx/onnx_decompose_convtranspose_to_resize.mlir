// Copyright (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx="convert-convtranspose-to-resize=true" %s -split-input-file | FileCheck %s

// Nearest-neighbor upsample, group=1 with block-diagonal all-ones weight.
// w[0,0] = ones, w[0,1] = zeros, w[1,0] = zeros, w[1,1] = ones.
func.func @convtranspose_nearest_group1(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
  %w = onnx.Constant dense<[[[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]]> : tensor<2x2x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x3x3xf32>, tensor<2x2x2x2xf32>, none) -> tensor<1x2x6x6xf32>
  onnx.Return %0 : tensor<1x2x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_nearest_group1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
// CHECK-DAG:       [[SCALES_:%.+]] = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
// CHECK:           [[RES_:%.+]] = "onnx.Resize"([[PARAM_0_]], %{{.*}}, [[SCALES_]], %{{.*}}) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<1x2x3x3xf32>, none, tensor<4xf32>, none) -> tensor<1x2x6x6xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<1x2x6x6xf32>
}

// -----

// Nearest-neighbor upsample, depthwise group=C with all-ones weight.
func.func @convtranspose_nearest_depthwise(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
  %w = onnx.Constant dense<1.000000e+00> : tensor<2x1x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>, none) -> tensor<1x2x6x6xf32>
  onnx.Return %0 : tensor<1x2x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_nearest_depthwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
// CHECK-DAG:       [[SCALES_:%.+]] = onnx.Constant dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>
// CHECK:           [[RES_:%.+]] = "onnx.Resize"([[PARAM_0_]], %{{.*}}, [[SCALES_]], %{{.*}}) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<1x2x3x3xf32>, none, tensor<4xf32>, none) -> tensor<1x2x6x6xf32>
// CHECK:           onnx.Return [[RES_]] : tensor<1x2x6x6xf32>
}

// -----

// Negative: group=1 weight is all-ones (not block-diagonal) -> off-diagonal
// channel blocks are non-zero, so it is NOT a per-channel replicator. Must stay
// a ConvTranspose.
func.func @convtranspose_not_blockdiag(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
  %w = onnx.Constant dense<1.000000e+00> : tensor<2x2x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x3x3xf32>, tensor<2x2x2x2xf32>, none) -> tensor<1x2x6x6xf32>
  onnx.Return %0 : tensor<1x2x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_not_blockdiag
// CHECK-NOT:       onnx.Resize
// CHECK:           "onnx.ConvTranspose"
}

// -----

// Negative: kernel == stride is required. Here strides=[1,1] so it is not an
// upsample; must stay a ConvTranspose.
func.func @convtranspose_not_upsample(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x4x4xf32> {
  %w = onnx.Constant dense<1.000000e+00> : tensor<2x1x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>, none) -> tensor<1x2x4x4xf32>
  onnx.Return %0 : tensor<1x2x4x4xf32>

// CHECK-LABEL:  func.func @convtranspose_not_upsample
// CHECK-NOT:       onnx.Resize
// CHECK:           "onnx.ConvTranspose"
}

// -----

// Negative: a general grouped ConvTranspose (1 < group < C_in) is neither the
// dense (group=1) nor the depthwise (group=C) per-channel-replicator encoding,
// so it is explicitly rejected and must stay a ConvTranspose - even though it
// is channel-preserving (C_out == C_in) with all-ones weights and kernel==stride.
func.func @convtranspose_grouped(%arg0: tensor<1x4x3x3xf32>) -> tensor<1x4x6x6xf32> {
  %w = onnx.Constant dense<1.000000e+00> : tensor<4x2x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x4x3x3xf32>, tensor<4x2x2x2xf32>, none) -> tensor<1x4x6x6xf32>
  onnx.Return %0 : tensor<1x4x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_grouped
// CHECK-NOT:       onnx.Resize
// CHECK:           "onnx.ConvTranspose"
}

// -----

// int8 QDQ, depthwise all-ones weight (int8 4 * 0.25 -> 1.0) with the bias
// behind a DequantizeLinear. The int32 bias is all-zero, so it dequantizes to
// 0 and the op is rewritten to onnx.Resize.
func.func @convtranspose_nearest_depthwise_qdq_bias_zero(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
  %w_i8 = onnx.Constant dense<4> : tensor<2x1x2x2xi8>
  %w_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %w_zp = onnx.Constant dense<0> : tensor<i8>
  %w = "onnx.DequantizeLinear"(%w_i8, %w_scale, %w_zp) {axis = 1 : si64} : (tensor<2x1x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<2x1x2x2xf32>
  %b_i32 = onnx.Constant dense<0> : tensor<2xi32>
  %b_scale = onnx.Constant dense<6.250000e-02> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<i32>
  %b = "onnx.DequantizeLinear"(%b_i32, %b_scale, %b_zp) {axis = 0 : si64} : (tensor<2xi32>, tensor<f32>, tensor<i32>) -> tensor<2xf32>
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x6x6xf32>
  onnx.Return %0 : tensor<1x2x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_nearest_depthwise_qdq_bias_zero
// CHECK:           "onnx.Resize"
// CHECK-NOT:       "onnx.ConvTranspose"
}

// -----

// Negative: same int8 QDQ depthwise upsample, but the int32 bias behind the
// DequantizeLinear is non-zero (8 * 0.0625 = 0.5), so it is not an all-zero
// bias and the op must stay a ConvTranspose.
func.func @convtranspose_nearest_depthwise_qdq_bias_nonzero(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
  %w_i8 = onnx.Constant dense<4> : tensor<2x1x2x2xi8>
  %w_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %w_zp = onnx.Constant dense<0> : tensor<i8>
  %w = "onnx.DequantizeLinear"(%w_i8, %w_scale, %w_zp) {axis = 1 : si64} : (tensor<2x1x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<2x1x2x2xf32>
  %b_i32 = onnx.Constant dense<8> : tensor<2xi32>
  %b_scale = onnx.Constant dense<6.250000e-02> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<i32>
  %b = "onnx.DequantizeLinear"(%b_i32, %b_scale, %b_zp) {axis = 0 : si64} : (tensor<2xi32>, tensor<f32>, tensor<i32>) -> tensor<2xf32>
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x6x6xf32>
  onnx.Return %0 : tensor<1x2x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_nearest_depthwise_qdq_bias_nonzero
// CHECK-NOT:       onnx.Resize
// CHECK:           "onnx.ConvTranspose"
}

// -----

// Negative (representability guard): int8 QDQ depthwise weight that is all-zero
// with scale = 1/256 and zero-point 0. The raw value that would dequantize to
// the all-ones target is 1 / (1/256) = 256, which is NOT representable in i8
// (range [-128, 127]). Narrowing 256 into i8 wraps to 0, which would spuriously
// match the all-zero stored weights - so without a range check this all-zero
// (all-zero-dequantized) weight would be misclassified as all-ones and wrongly
// rewritten to onnx.Resize. It must stay a ConvTranspose.
func.func @convtranspose_depthwise_qdq_weight_unrepresentable(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x6x6xf32> {
  %w_i8 = onnx.Constant dense<0> : tensor<2x1x2x2xi8>
  %w_scale = onnx.Constant dense<3.906250e-03> : tensor<f32>
  %w_zp = onnx.Constant dense<0> : tensor<i8>
  %w = "onnx.DequantizeLinear"(%w_i8, %w_scale, %w_zp) {axis = 1 : si64} : (tensor<2x1x2x2xi8>, tensor<f32>, tensor<i8>) -> tensor<2x1x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %w, %b) {auto_pad = "NOTSET", dilations = [1, 1], group = 2 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x3x3xf32>, tensor<2x1x2x2xf32>, none) -> tensor<1x2x6x6xf32>
  onnx.Return %0 : tensor<1x2x6x6xf32>

// CHECK-LABEL:  func.func @convtranspose_depthwise_qdq_weight_unrepresentable
// CHECK-NOT:       onnx.Resize
// CHECK:           "onnx.ConvTranspose"
}
