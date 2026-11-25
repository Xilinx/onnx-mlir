// RUN: onnx-mlir-opt --canonicalize-with-rn --recompose-onnx --canonicalize-with-rn %s --split-input-file | FileCheck %s

// Axis-1 normalization with reshaped moments and constant affine parameters.
func.func @reshaped_moments(%input: tensor<1x2048x64xf32>) -> tensor<1x512x256x1xf32> {
  %scale = onnx.Constant dense<2.0> : tensor<1x512x1xf32>
  %bias = onnx.Constant dense<0.5> : tensor<1x512x1xf32>
  %input_shape = onnx.Constant dense<[1, 512, 256]> : tensor<3xi64>
  %compact_shape = onnx.Constant dense<[1, 256]> : tensor<2xi64>
  %stat_shape = onnx.Constant dense<[1, 1, 256]> : tensor<3xi64>
  %output_shape = onnx.Constant dense<[1, 512, 256, 1]> : tensor<4xi64>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %epsilon = onnx.Constant dense<9.99999997E-7> : tensor<f32>
  %x = "onnx.Reshape"(%input, %input_shape) {allowzero = 0 : si64} : (tensor<1x2048x64xf32>, tensor<3xi64>) -> tensor<1x512x256xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %compact_second = "onnx.Reshape"(%second, %compact_shape) {allowzero = 0 : si64} : (tensor<1x1x256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %compact_mean = "onnx.Reshape"(%mean, %compact_shape) {allowzero = 0 : si64} : (tensor<1x1x256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
  %mean_square = "onnx.Mul"(%compact_mean, %compact_mean) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %variance = "onnx.Sub"(%compact_second, %mean_square) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %clamp = "onnx.Max"(%variance, %zero) : (tensor<1x256xf32>, tensor<f32>) -> tensor<1x256xf32>
  %ve = "onnx.Add"(%clamp, %epsilon) : (tensor<1x256xf32>, tensor<f32>) -> tensor<1x256xf32>
  %restored = "onnx.Reshape"(%ve, %stat_shape) {allowzero = 0 : si64} : (tensor<1x256xf32>, tensor<3xi64>) -> tensor<1x1x256xf32>
  %sqrt = "onnx.Sqrt"(%restored) : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %inv_std = "onnx.Reciprocal"(%sqrt) : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %centered = "onnx.Sub"(%x, %mean) : (tensor<1x512x256xf32>, tensor<1x1x256xf32>) -> tensor<1x512x256xf32>
  %normalized = "onnx.Mul"(%centered, %inv_std) : (tensor<1x512x256xf32>, tensor<1x1x256xf32>) -> tensor<1x512x256xf32>
  %scaled = "onnx.Mul"(%normalized, %scale) : (tensor<1x512x256xf32>, tensor<1x512x1xf32>) -> tensor<1x512x256xf32>
  %biased = "onnx.Add"(%scaled, %bias) : (tensor<1x512x256xf32>, tensor<1x512x1xf32>) -> tensor<1x512x256xf32>
  %output = "onnx.Reshape"(%biased, %output_shape) {allowzero = 0 : si64} : (tensor<1x512x256xf32>, tensor<4xi64>) -> tensor<1x512x256x1xf32>
  return %output : tensor<1x512x256x1xf32>
}


// CHECK-LABEL:  func.func @reshaped_moments(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x2048x64xf32>) -> tensor<1x512x256x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x512x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<1x512x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 512, 256]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[1, 512, 256, 1]> : tensor<4xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<1x2048x64xf32>, tensor<3xi64>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [0, 2, 1]} : (tensor<1x512x256xf32>) -> tensor<1x256x512xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [0, 2, 1]} : (tensor<1x512x1xf32>) -> tensor<1x1x512xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [0, 2, 1]} : (tensor<1x512x1xf32>) -> tensor<1x1x512xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_5_]], [[VAR_6_]], [[VAR_7_]]) {axis = 2 : si64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : si64} : (tensor<1x256x512xf32>, tensor<1x1x512xf32>, tensor<1x1x512xf32>) -> (tensor<1x256x512xf32>, none, none)
// CHECK:           [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_Y_]]) {perm = [0, 2, 1]} : (tensor<1x256x512xf32>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_8_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<1x512x256xf32>, tensor<4xi64>) -> tensor<1x512x256x1xf32>
// CHECK:           return [[VAR_9_]] : tensor<1x512x256x1xf32>
// CHECK:         }

// -----

// Single-element statistics with mixed ranks in the squared-mean branch.
func.func @singleton_moments(%x: tensor<1x512xf32>) -> tensor<1x512xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %epsilon = onnx.Constant dense<1.0e-6> : tensor<f32>
  %scale = onnx.Constant dense<2.0> : tensor<1x512xf32>
  %bias = onnx.Constant dense<0.5> : tensor<1x512xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %compact_mean = "onnx.Reshape"(%mean, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %product = "onnx.Mul"(%mean, %compact_mean) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %mean_square = "onnx.Reshape"(%product, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %clamp = "onnx.Max"(%variance, %zero) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %ve = "onnx.Add"(%clamp, %epsilon) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %sqrt = "onnx.Sqrt"(%ve) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %inv = "onnx.Reciprocal"(%sqrt) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %centered = "onnx.Sub"(%x, %mean) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
  %normalized = "onnx.Mul"(%centered, %inv) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
  %scaled = "onnx.Mul"(%normalized, %scale) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %biased = "onnx.Add"(%scaled, %bias) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  return %biased : tensor<1x512xf32>
}

// CHECK-LABEL:  func.func @singleton_moments(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512xf32>) -> tensor<1x512xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x512xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<1x512xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {axis = 1 : si64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : si64} : (tensor<1x512xf32>, tensor<1x512xf32>, tensor<1x512xf32>) -> (tensor<1x512xf32>, none, none)
// CHECK:           return [[VAR_Y_]] : tensor<1x512xf32>
// CHECK:         }

// -----

// A live shared mean selects RMSNorm over the existing centered input.
func.func @shared_mean(%x: tensor<1x512xf32>) -> (tensor<1x512xf32>, tensor<1x1xf32>) {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %epsilon = onnx.Constant dense<1.0e-6> : tensor<f32>
  %scale = onnx.Constant dense<2.0> : tensor<1x512xf32>
  %bias = onnx.Constant dense<0.5> : tensor<1x512xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %compact_mean = "onnx.Reshape"(%mean, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %product = "onnx.Mul"(%mean, %compact_mean) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %mean_square = "onnx.Reshape"(%product, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %clamp = "onnx.Max"(%variance, %zero) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %ve = "onnx.Add"(%clamp, %epsilon) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %sqrt = "onnx.Sqrt"(%ve) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %inv = "onnx.Reciprocal"(%sqrt) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %centered = "onnx.Sub"(%x, %mean) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
  %normalized = "onnx.Mul"(%centered, %inv) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
  %scaled = "onnx.Mul"(%normalized, %scale) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %biased = "onnx.Add"(%scaled, %bias) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  return %biased, %mean : tensor<1x512xf32>, tensor<1x1xf32>
}

// CHECK-LABEL:  func.func @shared_mean(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512xf32>) -> (tensor<1x512xf32>, tensor<1x1xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x512xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<1x512xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_3_]]) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_4_]], [[VAR_1_]], [[VAR_2_]]) {axis = 1 : si64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : si64} : (tensor<1x512xf32>, tensor<1x512xf32>, tensor<1x512xf32>) -> (tensor<1x512xf32>, none)
// CHECK:           return [[VAR_Y_]], [[VAR_3_]] : tensor<1x512xf32>, tensor<1x1xf32>
// CHECK:         }

// -----

// Local variance cleanup preserves a squared mean with another user.
func.func @shared_mean_square(%x: tensor<1x512xf32>) -> (tensor<1x512xf32>, tensor<1xf32>) {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %epsilon = onnx.Constant dense<1.0e-6> : tensor<f32>
  %scale = onnx.Constant dense<2.0> : tensor<1x512xf32>
  %bias = onnx.Constant dense<0.5> : tensor<1x512xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %compact_mean = "onnx.Reshape"(%mean, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %product = "onnx.Mul"(%mean, %compact_mean) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %mean_square = "onnx.Reshape"(%product, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  %clamp = "onnx.Max"(%variance, %zero) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %ve = "onnx.Add"(%clamp, %epsilon) : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %sqrt = "onnx.Sqrt"(%ve) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %inv = "onnx.Reciprocal"(%sqrt) : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %centered = "onnx.Sub"(%x, %mean) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
  %normalized = "onnx.Mul"(%centered, %inv) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
  %scaled = "onnx.Mul"(%normalized, %scale) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  %biased = "onnx.Add"(%scaled, %bias) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
  return %biased, %mean_square : tensor<1x512xf32>, tensor<1xf32>
}

// CHECK-LABEL:  func.func @shared_mean_square(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512xf32>) -> (tensor<1x512xf32>, tensor<1xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<1x512xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<1x512xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512xf32>, tensor<1xi64>) -> tensor<1x1xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_3_]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_3_]]) : (tensor<1x512xf32>, tensor<1x1xf32>) -> tensor<1x512xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_6_]], [[VAR_1_]], [[VAR_2_]]) {axis = 1 : si64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : si64} : (tensor<1x512xf32>, tensor<1x512xf32>, tensor<1x512xf32>) -> (tensor<1x512xf32>, none)
// CHECK:           return [[VAR_Y_]], [[VAR_5_]] : tensor<1x512xf32>, tensor<1xf32>
// CHECK:         }
