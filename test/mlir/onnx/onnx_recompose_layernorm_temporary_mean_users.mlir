// RUN: onnx-mlir-opt --onnx-hybrid-transform="decomposition=false" --canonicalize-with-rn %s | FileCheck %s

// Temporary squared-mean users must not force an RMSNorm fallback.
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
