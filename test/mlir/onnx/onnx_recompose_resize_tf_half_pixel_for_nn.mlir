// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt --recompose-onnx -split-input-file %s --verify-diagnostics | FileCheck %s

// tf_half_pixel_for_nn + nearest + floor + integer upsampling scale is exactly
// equivalent to asymmetric, so it is rewritten to asymmetric (AIESW-46580).
func.func @resize_tf_half_pixel_for_nn_nearest_floor_2x(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %roi) {
    antialias = 0 : si64,
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    cubic_coeff_a = -7.500000e-01 : f32,
    exclude_outside = 0 : si64,
    extrapolation_value = 0.000000e+00 : f32,
    keep_aspect_ratio_policy = "stretch",
    mode = "nearest",
    nearest_mode = "floor"} : (tensor<1x64x16x16xf32>, none, tensor<4xf32>, none) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_nearest_floor_2x
// CHECK: "onnx.Resize"
// CHECK-SAME: coordinate_transformation_mode = "asymmetric"
// CHECK-SAME: mode = "nearest"
// CHECK-SAME: nearest_mode = "floor"

// -----

// sizes path (no scales): integer shape ratio is enough to rewrite.
func.func @resize_tf_half_pixel_for_nn_via_sizes(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %sizes = onnx.Constant dense<[1, 64, 32, 32]> : tensor<4xi64>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %roi, %sizes) {
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    mode = "nearest",
    nearest_mode = "floor"} : (tensor<1x64x16x16xf32>, none, none, tensor<4xi64>) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_via_sizes
// CHECK: coordinate_transformation_mode = "asymmetric"

// -----

// round_prefer_floor is NOT floor-safe: it rounds and breaks on ties (at 2x,
// out coord 1 maps tf 0.75 -> 1 but asym 0.5 -> 0), so it is left unchanged.
func.func @resize_tf_half_pixel_for_nn_round_prefer_floor(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %roi) {
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    mode = "nearest",
    nearest_mode = "round_prefer_floor"} : (tensor<1x64x16x16xf32>, none, tensor<4xf32>, none) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_round_prefer_floor
// CHECK: coordinate_transformation_mode = "tf_half_pixel_for_nn"

// -----

// linear mode is NOT rewritten: the offset is not floored, so the mapping is
// observable and the equivalence does not hold.
func.func @resize_tf_half_pixel_for_nn_linear_unchanged(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %roi) {
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    mode = "linear",
    nearest_mode = "floor"} : (tensor<1x64x16x16xf32>, none, tensor<4xf32>, none) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_linear_unchanged
// CHECK: coordinate_transformation_mode = "tf_half_pixel_for_nn"

// -----

// round_prefer_ceil is NOT floor-safe and is left unchanged.
func.func @resize_tf_half_pixel_for_nn_round_prefer_ceil_unchanged(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x32x32xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %roi) {
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    mode = "nearest",
    nearest_mode = "round_prefer_ceil"} : (tensor<1x64x16x16xf32>, none, tensor<4xf32>, none) -> tensor<1x64x32x32xf32>
  return %0 : tensor<1x64x32x32xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_round_prefer_ceil_unchanged
// CHECK: coordinate_transformation_mode = "tf_half_pixel_for_nn"

// -----

// Non-integer scale (16 -> 24 is 1.5x) is NOT rewritten: the 0.5/s offset can
// cross an integer boundary, so the mapping is observable.
func.func @resize_tf_half_pixel_for_nn_non_integer_scale_unchanged(%arg0: tensor<1x64x16x16xf32>) -> tensor<1x64x24x24xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 1.5, 1.5]> : tensor<4xf32>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %roi) {
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    mode = "nearest",
    nearest_mode = "floor"} : (tensor<1x64x16x16xf32>, none, tensor<4xf32>, none) -> tensor<1x64x24x24xf32>
  return %0 : tensor<1x64x24x24xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_non_integer_scale_unchanged
// CHECK: coordinate_transformation_mode = "tf_half_pixel_for_nn"

// -----

// Non-integer scales whose shape ratio looks integer (in=4, scale=2.1, out=8)
// must NOT be rewritten: the coordinate transform uses the scale value, and
// floor((2+0.5)/2.1)=1 != floor(2/2.1)=0.
func.func @resize_tf_half_pixel_for_nn_non_integer_scale_integer_shape_unchanged(%arg0: tensor<1x64x4x4xf32>) -> tensor<1x64x8x8xf32> {
  %roi = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.1, 2.1]> : tensor<4xf32>
  // expected-warning@+1 {{'tf_half_pixel_for_nn' is only supported in opset 11.}}
  %0 = "onnx.Resize"(%arg0, %roi, %scales, %roi) {
    coordinate_transformation_mode = "tf_half_pixel_for_nn",
    mode = "nearest",
    nearest_mode = "floor"} : (tensor<1x64x4x4xf32>, none, tensor<4xf32>, none) -> tensor<1x64x8x8xf32>
  return %0 : tensor<1x64x8x8xf32>
}
// CHECK-LABEL: func.func @resize_tf_half_pixel_for_nn_non_integer_scale_integer_shape_unchanged
// CHECK: coordinate_transformation_mode = "tf_half_pixel_for_nn"
