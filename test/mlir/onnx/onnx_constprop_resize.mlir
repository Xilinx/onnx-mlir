// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --shape-inference --constprop-onnx %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Constant folding of onnx.Resize (ResizeofConst pattern).
//
// A Resize whose data + sizes/scales inputs are all constant folds to a single
// onnx.Constant.
//
// The negative cases at the bottom exercise the isResizeConstPropagatable gate:
// an unsupported configuration must be left untouched (onnx.Resize survives).
//===----------------------------------------------------------------------===//

// cubic / half_pixel via `sizes`, downsample 4x4 -> 2x2.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     3.03125   5.21875
//     5  6  7  8       =>           11.78125  13.96875
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_cubic_half_pixel_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_cubic_half_pixel_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}3.031250e+00, 5.218750e+00], [11.78125, 13.96875]]]]> : tensor<1x1x2x2xf32>
}

// -----

// linear / half_pixel via `sizes`, downsample 4x4 -> 2x2.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     3.5   5.5
//     5  6  7  8       =>           11.5  13.5
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_linear_half_pixel_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_linear_half_pixel_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}3.500000e+00, 5.500000e+00], [1.150000e+01, 1.350000e+01]]]]> : tensor<1x1x2x2xf32>
}

// -----

// linear / align_corners via `scales`, upsample 2x2 -> 4x4.
//
//    Input (2x2)          Output (4x4)
//     1  2                 1.000  1.333  1.667  2.000
//     3  4        =>       1.667  2.000  2.333  2.667
//                          2.333  2.667  3.000  3.333
//                          3.000  3.333  3.667  4.000
//
// CHECK-LABEL: @resize_linear_align_corners_scales() -> tensor<1x1x4x4xf32>
func.func @resize_linear_align_corners_scales() -> tensor<1x1x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>} : () -> tensor<1x1x2x2xf32>
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %r = "onnx.Resize"(%data, %none, %scales, %none) {antialias = 0 : si64, coordinate_transformation_mode = "align_corners", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xf32>
  onnx.Return %r : tensor<1x1x4x4xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+00, 1.33333337, 1.66666663, 2.000000e+00], [1.66666663, 2.000000e+00, 2.33333325, 2.66666675], [2.33333325, 2.66666675, 3.000000e+00, 3.33333325], [3.000000e+00, 3.33333325, 3.66666675, 4.000000e+00]]]]> : tensor<1x1x4x4xf32>
}

// -----

// linear / asymmetric via `sizes`, downsample 4x4 -> 2x2.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     1   3
//     5  6  7  8       =>            9  11
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_linear_asymmetric_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_linear_asymmetric_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+00, 3.000000e+00], [9.000000e+00, 1.100000e+01]]]]> : tensor<1x1x2x2xf32>
}

// -----

// nearest / asymmetric / floor via `sizes`, downsample 4x4 -> 2x2.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     1   3
//     5  6  7  8       =>            9  11
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_nearest_asymmetric_floor_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_nearest_asymmetric_floor_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+00, 3.000000e+00], [9.000000e+00, 1.100000e+01]]]]> : tensor<1x1x2x2xf32>
}

// -----

// nearest / half_pixel / round_prefer_ceil via `scales`, upsample 2x2 -> 4x4.
//
//    Input (2x2)          Output (4x4)
//     1  2                 1  1  2  2
//     3  4        =>       1  1  2  2
//                          3  3  4  4
//                          3  3  4  4
//
// CHECK-LABEL: @resize_nearest_half_pixel_round_prefer_ceil_scales() -> tensor<1x1x4x4xf32>
func.func @resize_nearest_half_pixel_round_prefer_ceil_scales() -> tensor<1x1x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>} : () -> tensor<1x1x2x2xf32>
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %r = "onnx.Resize"(%data, %none, %scales, %none) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "round_prefer_ceil"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xf32>
  onnx.Return %r : tensor<1x1x4x4xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00], [3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00]]]]> : tensor<1x1x4x4xf32>
}

// -----

// cubic / half_pixel with exclude_outside=1 via `sizes`, downsample 4x4 -> 2x2.
// Border taps that fall outside are dropped (weight 0) and the remaining
// weights renormalized, so the values differ from the exclude_outside=0 case.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     2.85714   5.11429
//     5  6  7  8       =>           11.88571  14.14286
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_cubic_half_pixel_exclude_outside_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_cubic_half_pixel_exclude_outside_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 1 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}2.85714293, 5.11428595], [11.8857145, 14.1428576]]]]> : tensor<1x1x2x2xf32>
}

// -----

// cubic / half_pixel with a non-default cubic_coeff_a (-0.5) via `sizes`,
// downsample 4x4 -> 2x2. Confirms cubic_coeff_a is plumbed into the kernel
// (different curve => different values than the -0.75 default case above).
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     3.1875   5.3125
//     5  6  7  8       =>           11.6875  13.8125
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_cubic_half_pixel_cubic_coeff_a_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_cubic_half_pixel_cubic_coeff_a_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -5.000000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}3.187500e+00, 5.312500e+00], [1.168750e+01, 1.381250e+01]]]]> : tensor<1x1x2x2xf32>
}

// -----

// linear / pytorch_half_pixel via `sizes`, downsample 4x4 -> 2x2.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     3.5   5.5
//     5  6  7  8       =>           11.5  13.5
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_linear_pytorch_half_pixel_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_linear_pytorch_half_pixel_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}3.500000e+00, 5.500000e+00], [1.150000e+01, 1.350000e+01]]]]> : tensor<1x1x2x2xf32>
}

// -----

// linear / half_pixel_symmetric via `sizes`, downsample 4x4 -> 2x2.
//
//    Input (4x4)                    Output (2x2)
//     1  2  3  4                     3.5   5.5
//     5  6  7  8       =>           11.5  13.5
//     9 10 11 12
//    13 14 15 16
//
// CHECK-LABEL: @resize_linear_half_pixel_symmetric_sizes() -> tensor<1x1x2x2xf32>
func.func @resize_linear_half_pixel_symmetric_sizes() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel_symmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}3.500000e+00, 5.500000e+00], [1.150000e+01, 1.350000e+01]]]]> : tensor<1x1x2x2xf32>
}

// -----

// nearest / asymmetric / ceil via `scales`, upsample 2x2 -> 4x4.
//
//    Input (2x2)          Output (4x4)
//     1  2                 1  2  2  2
//     3  4        =>       3  4  4  4
//                          3  4  4  4
//                          3  4  4  4
//
// CHECK-LABEL: @resize_nearest_asymmetric_ceil_scales() -> tensor<1x1x4x4xf32>
func.func @resize_nearest_asymmetric_ceil_scales() -> tensor<1x1x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>} : () -> tensor<1x1x2x2xf32>
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %r = "onnx.Resize"(%data, %none, %scales, %none) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "ceil"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xf32>
  onnx.Return %r : tensor<1x1x4x4xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00], [3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00], [3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]]]]> : tensor<1x1x4x4xf32>
}

// -----

// nearest / half_pixel / round_prefer_floor via `scales`, upsample 2x2 -> 4x4.
//
//    Input (2x2)          Output (4x4)
//     1  2                 1  1  2  2
//     3  4        =>       1  1  2  2
//                          3  3  4  4
//                          3  3  4  4
//
// CHECK-LABEL: @resize_nearest_half_pixel_round_prefer_floor_scales() -> tensor<1x1x4x4xf32>
func.func @resize_nearest_half_pixel_round_prefer_floor_scales() -> tensor<1x1x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>} : () -> tensor<1x1x2x2xf32>
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %r = "onnx.Resize"(%data, %none, %scales, %none) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xf32>
  onnx.Return %r : tensor<1x1x4x4xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00], [3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00]]]]> : tensor<1x1x4x4xf32>
}

// -----

// Single-axis resize: only W changes (H unchanged), linear / half_pixel via
// `sizes`, 2x4 -> 2x2. Exercises the per-axis 'continue' skip (H is untouched)
// and a lone 1-D interpolation pass along the innermost axis.
//
//    Input (2x4)                Output (2x2)
//     1  2  3  4      =>          1.5  3.5
//     5  6  7  8                  5.5  7.5
//
// CHECK-LABEL: @resize_linear_half_pixel_single_axis_width() -> tensor<1x1x2x2xf32>
func.func @resize_linear_half_pixel_single_axis_width() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]> : tensor<1x1x2x4xf32>} : () -> tensor<1x1x2x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x2x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.500000e+00, 3.500000e+00], [5.500000e+00, 7.500000e+00]]]]> : tensor<1x1x2x2xf32>
}

// -----

// cubic / half_pixel upsample of a step edge, width 4 -> 8. This shows why cubic
// uses four input pixels per output pixel: near the 0->10 jump the fitted cubic
// overshoots the input range, producing a slightly negative value (-1.05) just
// before the edge and a value above 10 (11.05) just after it. That over/under-
// shoot is the signature of the 4-tap kernel; a 2-tap linear resize (shown for
// contrast) can only stay within [0, 10] and never overshoots.
//
//    Input (1x4):   0    0    10   10
//
//    cubic  (1x8):  0  -0.35  -1.05  2.27  7.73  11.05  10.35  10   <- overshoots
//    linear (1x8):  0   0      0     2.5   7.5   10      10     10   <- clamped
//
// CHECK-LABEL: @resize_cubic_half_pixel_upscale_step() -> tensor<1x1x1x8xf32>
func.func @resize_cubic_half_pixel_upscale_step() -> tensor<1x1x1x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[0.0, 0.0, 10.0, 10.0]]]]> : tensor<1x1x1x4xf32>} : () -> tensor<1x1x1x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 1, 8]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "floor"} : (tensor<1x1x1x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x1x8xf32>
  onnx.Return %r : tensor<1x1x1x8xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}0.000000e+00, -0.3515625, -1.0546875, 2.265625, 7.734375, 11.0546875, 10.3515625, 1.000000e+01]]]]> : tensor<1x1x1x8xf32>
}

// -----

// Non-integer scale (1.1) via `scales` where the output length equals the input
// length: outLen = floor(inLen * scale) = floor(4 * 1.1) = 4. The lengths match
// but the axis must STILL be resampled, because half_pixel maps the output
// coordinates to fractional input positions (-0.05, 0.86, 1.77, 2.68), not to
// the identity 0, 1, 2, 3. Guards against skipping an axis on length-equality
// alone (the skip must require scale == 1.0).
//
//    Input (1x4):  10  20  30  40
//    Output (1x4): 10  18.64  27.73  36.82     (linear / half_pixel, scale 1.1)
//
// CHECK-LABEL: @resize_linear_half_pixel_scale_gt1_equal_len() -> tensor<1x1x1x4xf32>
func.func @resize_linear_half_pixel_scale_gt1_equal_len() -> tensor<1x1x1x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[10.0, 20.0, 30.0, 40.0]]]]> : tensor<1x1x1x4xf32>} : () -> tensor<1x1x1x4xf32>
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 1.0, 1.1]> : tensor<4xf32>} : () -> tensor<4xf32>
  %r = "onnx.Resize"(%data, %none, %scales, %none) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x1x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x1x4xf32>
  onnx.Return %r : tensor<1x1x1x4xf32>
  // CHECK-NOT: onnx.Resize
  // CHECK: onnx.Constant dense<{{\[\[}}{{\[\[}}1.000000e+01, 18.636364, 27.727272, 36.8181801]]]]> : tensor<1x1x1x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative cases: isResizeConstPropagatable gate must leave onnx.Resize intact.
//===----------------------------------------------------------------------===//

// Non-constant data input: the pattern must not fire.
// CHECK-LABEL: @resize_nonconst_data
func.func @resize_nonconst_data(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%arg0, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK: onnx.Resize
}

// -----

// roi present (tf_crop_and_resize): unsupported, gated out.
// CHECK-LABEL: @resize_roi_present
func.func @resize_roi_present() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %roi = "onnx.Constant"() {value = dense<[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]> : tensor<8xf32>} : () -> tensor<8xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %roi, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "tf_crop_and_resize", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, tensor<8xf32>, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK: onnx.Resize
}

// -----

// Integer element type: float-only fold, gated out.
// CHECK-LABEL: @resize_integer_dtype
func.func @resize_integer_dtype() -> tensor<1x1x2x2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]> : tensor<1x1x4x4xi32>} : () -> tensor<1x1x4x4xi32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<1x1x4x4xi32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xi32>
  onnx.Return %r : tensor<1x1x2x2xi32>
  // CHECK: onnx.Resize
}

// -----

// antialias = 1: unsupported, gated out.
// CHECK-LABEL: @resize_antialias_enabled
func.func @resize_antialias_enabled() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 2, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 1 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<4xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK: onnx.Resize
}

// -----

// axes attribute present (partial-axis resize): unsupported, gated out.
// CHECK-LABEL: @resize_axes_attr
func.func @resize_axes_attr() -> tensor<1x1x2x2xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %data = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]]> : tensor<1x1x4x4xf32>} : () -> tensor<1x1x4x4xf32>
  %sizes = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %r = "onnx.Resize"(%data, %none, %none, %sizes) {antialias = 0 : si64, axes = [2, 3], coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "cubic", nearest_mode = "floor"} : (tensor<1x1x4x4xf32>, none, none, tensor<2xi64>) -> tensor<1x1x2x2xf32>
  onnx.Return %r : tensor<1x1x2x2xf32>
  // CHECK: onnx.Resize
}
