// RUN: onnx-mlir-opt %s --split-input-file --quant-types --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @float_input_quantized_affine_qdq
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i16:f32,
// CHECK-SAME: tensor<4x!quant.uniform<i16:f32,
// CHECK-SAME: -> (tensor<1x2x4xf32>, none, none)
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: return
func.func @float_input_quantized_affine_qdq(%input: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
  %scale_storage = onnx.Constant dense<2> : tensor<4xi16>
  %bias_storage = onnx.Constant dense<1> : tensor<4xi16>
  %scale_step = onnx.Constant dense<0.5> : tensor<f32>
  %bias_step = onnx.Constant dense<0.25> : tensor<f32>
  %zero = onnx.Constant dense<0> : tensor<i16>
  %scale = "onnx.DequantizeLinear"(%scale_storage, %scale_step, %zero) : (tensor<4xi16>, tensor<f32>, tensor<i16>) -> tensor<4xf32>
  %bias = "onnx.DequantizeLinear"(%bias_storage, %bias_step, %zero) : (tensor<4xi16>, tensor<f32>, tensor<i16>) -> tensor<4xf32>
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @quantized_input_float_affine_qdq
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: "onnx.LayerNormalization"
// CHECK-SAME: (tensor<1x2x4x!quant.uniform<i16:f32,
// CHECK-SAME: tensor<4xf32>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: return
func.func @quantized_input_float_affine_qdq(%storage: tensor<1x2x4xi16>, %scale: tensor<4xf32>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  %step = onnx.Constant dense<0.25> : tensor<f32>
  %zero = onnx.Constant dense<0> : tensor<i16>
  %input = "onnx.DequantizeLinear"(%storage, %step, %zero) : (tensor<1x2x4xi16>, tensor<f32>, tensor<i16>) -> tensor<1x2x4xf32>
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @rms_float_input_quantized_scale_qdq
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: "onnx.RMSLayerNormalization"
// CHECK-SAME: (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i16:f32,
// CHECK-SAME: none) -> (tensor<1x2x4xf32>, none)
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: return
func.func @rms_float_input_quantized_scale_qdq(%input: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
  %scale_storage = onnx.Constant dense<2> : tensor<4xi16>
  %step = onnx.Constant dense<0.5> : tensor<f32>
  %zero = onnx.Constant dense<0> : tensor<i16>
  %scale = "onnx.DequantizeLinear"(%scale_storage, %step, %zero) : (tensor<4xi16>, tensor<f32>, tensor<i16>) -> tensor<4xf32>
  %absent = "onnx.NoValue"() {value} : () -> none
  %output, %invstd = "onnx.RMSLayerNormalization"(%input, %scale, %absent) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, none) -> (tensor<1x2x4xf32>, none)
  return %output : tensor<1x2x4xf32>
}
