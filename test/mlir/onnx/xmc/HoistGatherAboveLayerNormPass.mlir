// RUN: onnx-mlir-opt --split-input-file --hoist-gather-above-layernorm %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// HoistGatherAboveLayerNormPass: dq0/LN/q0/dq1/q1 are updated in place where
// possible; a new Gather is inserted before dq0 (or before LN for pattern B).
//
// Pattern (A): dq0 -> LayerNorm -> q0 -> dq1 -> Gather -> q1
//   => Gather -> dq0 -> LayerNorm -> q0 -> dq1 -> q1
// Pattern (B): LayerNorm -> Gather => Gather -> LayerNorm

// CHECK-LABEL: @pooler_qdq_gather_chain
// CHECK:       "onnx.Gather"({{.*}}) {axis = 1 : si64, onnx_node_name = "/pooler/Gather"
// CHECK-SAME:  : (tensor<1x64x768xui16>, tensor<i64>) -> tensor<1x768xui16>
// CHECK:       "onnx.DequantizeLinear"{{.*}}onnx_node_name = "/model/input_dequant"
// CHECK-SAME:  : (tensor<1x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x768xf32>
// CHECK:       "onnx.LayerNormalization"{{.*}}onnx_node_name = "/model/layer_norm"
// CHECK-SAME:  : (tensor<1x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x768xf32>, none, none)
// CHECK:       "onnx.QuantizeLinear"{{.*}}onnx_node_name = "/model/layer_norm_output_quant"
// CHECK-SAME:  : (tensor<1x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x768xui16>
// CHECK:       "onnx.DequantizeLinear"{{.*}} : (tensor<1x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x768xf32>
// CHECK:       "onnx.QuantizeLinear"{{.*}} : (tensor<1x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x768xui16>
// CHECK-NOT:   tensor<1x64x768xf32>
// CHECK-NOT:   "onnx.Gather"({{.*}}) : (tensor<1x64x768xf32>
// CHECK:       return
func.func @pooler_qdq_gather_chain(%arg0: tensor<1x64x768xf32>) -> tensor<1x768xui16> {
  %idx = onnx.Constant dense<0> : tensor<i64>
  %zp_in = onnx.Constant dense<30800> : tensor<ui16>
  %scale_in = onnx.Constant dense<6.42855535E-4> : tensor<f32>
  %scale_ln = onnx.Constant dense<4.20910917E-4> : tensor<f32>
  %zp_ln = onnx.Constant dense<29511> : tensor<ui16>
  %weight = onnx.Constant dense<1.0> : tensor<768xf32>
  %bias = onnx.Constant dense<0.0> : tensor<768xf32>

  %q_in = "onnx.QuantizeLinear"(%arg0, %scale_in, %zp_in) {
    axis = 1 : si64, output_dtype = 0 : si64, saturate = 1 : si64
  } : (tensor<1x64x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xui16>
  %dq0 = "onnx.DequantizeLinear"(%q_in, %scale_in, %zp_in) {
    axis = 1 : si64, onnx_node_name = "/model/input_dequant"
  } : (tensor<1x64x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xf32>
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%dq0, %weight, %bias) {
    axis = -1 : si64, epsilon = 1.000000e-07 : f32, stash_type = 1 : si64,
    onnx_node_name = "/model/layer_norm"
  } : (tensor<1x64x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x64x768xf32>, none, none)
  %q0 = "onnx.QuantizeLinear"(%Y, %scale_ln, %zp_ln) {
    axis = 1 : si64, output_dtype = 0 : si64, saturate = 1 : si64,
    onnx_node_name = "/model/layer_norm_output_quant"
  } : (tensor<1x64x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xui16>
  %dq1 = "onnx.DequantizeLinear"(%q0, %scale_ln, %zp_ln) {
    axis = 1 : si64
  } : (tensor<1x64x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x64x768xf32>
  %gather_out = "onnx.Gather"(%dq1, %idx) {
    axis = 1 : si64, onnx_node_name = "/pooler/Gather"
  } : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x768xf32>
  %q1 = "onnx.QuantizeLinear"(%gather_out, %scale_ln, %zp_ln) {
    axis = 1 : si64, output_dtype = 0 : si64, saturate = 1 : si64
  } : (tensor<1x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x768xui16>
  return %q1 : tensor<1x768xui16>
}

// -----

// Pattern (B): in-place LayerNorm; old Gather removed (consumer uses LN.Y).

// CHECK-LABEL: @layernorm_to_gather_f32
// CHECK:       "onnx.Gather"({{.*}}) {axis = 1 : si64} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x768xf32>
// CHECK-NEXT:  %{{.*}}, %{{.*}}, %{{.*}} = "onnx.LayerNormalization"{{.*}} : (tensor<1x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x768xf32>, none, none)
// CHECK-NOT:   "onnx.Gather"
// CHECK:       return %{{.*}} : tensor<1x768xf32>
func.func @layernorm_to_gather_f32(%arg0: tensor<1x64x768xf32>) -> tensor<1x768xf32> {
  %idx = onnx.Constant dense<0> : tensor<i64>
  %weight = onnx.Constant dense<1.0> : tensor<768xf32>
  %bias = onnx.Constant dense<0.0> : tensor<768xf32>
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %weight, %bias) {
    axis = -1 : si64, epsilon = 1.000000e-07 : f32, stash_type = 1 : si64
  } : (tensor<1x64x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x64x768xf32>, none, none)
  %out = "onnx.Gather"(%Y, %idx) {axis = 1 : si64} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x768xf32>
  return %out : tensor<1x768xf32>
}

// -----

// Scalar index drops a dimension: LayerNorm axis must be adjusted (2 -> 1).

// CHECK-LABEL: @scalar_index_adjusts_layernorm_axis
// CHECK:       "onnx.Gather"({{.*}}) {axis = 1 : si64} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x768xf32>
// CHECK:       "onnx.LayerNormalization"{{.*}} {axis = 1 : si64
// CHECK-SAME:  : (tensor<1x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x768xf32>, none, none)
// CHECK-NOT:   axis = 2 : si64
func.func @scalar_index_adjusts_layernorm_axis(%arg0: tensor<1x64x768xf32>) -> tensor<1x768xf32> {
  %idx = onnx.Constant dense<0> : tensor<i64>
  %weight = onnx.Constant dense<1.0> : tensor<768xf32>
  %bias = onnx.Constant dense<0.0> : tensor<768xf32>
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %weight, %bias) {
    axis = 2 : si64, epsilon = 1.000000e-07 : f32, stash_type = 1 : si64
  } : (tensor<1x64x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x64x768xf32>, none, none)
  %out = "onnx.Gather"(%Y, %idx) {axis = 1 : si64} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x768xf32>
  return %out : tensor<1x768xf32>
}

// -----

// LN axis (-1 -> 2) is not greater than Gather axis (2): no hoist.

// CHECK-LABEL: @gather_axis_equals_ln_axis_no_change
// CHECK:       "onnx.LayerNormalization"{{.*}} : (tensor<1x64x768xf32>
// CHECK:       "onnx.Gather"({{.*}}) {axis = 2 : si64} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x64x768xf32>
// CHECK-NOT:   "onnx.Gather"({{.*}}) : (tensor<1x64x768xui16>
func.func @gather_axis_equals_ln_axis_no_change(%arg0: tensor<1x64x768xf32>) -> tensor<1x64x768xf32> {
  %idx = onnx.Constant dense<0> : tensor<i64>
  %weight = onnx.Constant dense<1.0> : tensor<768xf32>
  %bias = onnx.Constant dense<0.0> : tensor<768xf32>
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %weight, %bias) {
    axis = -1 : si64, epsilon = 1.000000e-07 : f32, stash_type = 1 : si64
  } : (tensor<1x64x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x64x768xf32>, none, none)
  %out = "onnx.Gather"(%Y, %idx) {axis = 2 : si64} : (tensor<1x64x768xf32>, tensor<i64>) -> tensor<1x64x768xf32>
  return %out : tensor<1x64x768xf32>
}

// -----

// Gather that preserves tensor size must not hoist.

// CHECK-LABEL: @gather_same_size_no_change
// CHECK:       "onnx.LayerNormalization"{{.*}} : (tensor<1x64x768xf32>
// CHECK:       "onnx.Gather"({{.*}}) {axis = 1 : si64} : (tensor<1x64x768xf32>, tensor<64xi64>) -> tensor<1x64x768xf32>
// CHECK-NOT:   "onnx.Gather"({{.*}}) : (tensor<1x64x768xf32>, tensor<64xi64>)
func.func @gather_same_size_no_change(%arg0: tensor<1x64x768xf32>) -> tensor<1x64x768xf32> {
  %idx = onnx.Constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
  %weight = onnx.Constant dense<1.0> : tensor<768xf32>
  %bias = onnx.Constant dense<0.0> : tensor<768xf32>
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %weight, %bias) {
    axis = -1 : si64, epsilon = 1.000000e-07 : f32, stash_type = 1 : si64
  } : (tensor<1x64x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x64x768xf32>, none, none)
  %out = "onnx.Gather"(%Y, %idx) {axis = 1 : si64} : (tensor<1x64x768xf32>, tensor<64xi64>) -> tensor<1x64x768xf32>
  return %out : tensor<1x64x768xf32>
}
