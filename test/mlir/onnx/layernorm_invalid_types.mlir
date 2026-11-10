// RUN: onnx-mlir-opt %s --split-input-file --verify-diagnostics -o /dev/null
// RUN: not onnx-mlir-opt %s --split-input-file -o /dev/null 2>&1 | FileCheck %s
// CHECK: LayerNormalization op with incompatible B type

func.func @incompatible_float_bias(%input: tensor<1x2x4xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf16>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible B type}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, tensor<4xf16>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @incompatible_float_scale(%input: tensor<1x2x4xf32>, %scale: tensor<4xf16>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible scale type}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf16>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @incompatible_quantized_bias(%input: tensor<1x2x4xf32>, %scale: tensor<4xf32>, %bias: tensor<4x!quant.uniform<i16:f16, 1.0>>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible B type}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, tensor<4x!quant.uniform<i16:f16, 1.0>>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @incompatible_quantized_input(%input: tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, %scale: tensor<4xf32>, %bias: tensor<4xf16>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible B type}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, tensor<4xf32>, tensor<4xf16>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @incompatible_per_axis_scale(%input: tensor<1x2x4xf32>, %scale: tensor<4x!quant.uniform<i8:f16:0, {0.25, 0.5, 0.75, 1.0}>>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible scale type}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i8:f16:0, {0.25, 0.5, 0.75, 1.0}>>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @incompatible_bias_shape(%input: tensor<1x2x4xf32>, %scale: tensor<4xf32>, %bias: tensor<3x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible B shapes (broadcast)}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, tensor<3x!quant.uniform<i16:f32, 1.0>>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @incompatible_scale_rank(%input: tensor<1x2x4xf32>, %scale: tensor<1x1x2x4x!quant.uniform<i16:f32, 1.0>>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible scale shapes (unidirectional broadcast)}}
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<1x1x2x4x!quant.uniform<i16:f32, 1.0>>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @rms_incompatible_quantized_input(%input: tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, %scale: tensor<4xf16>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  // expected-error @+1 {{LayerNormalization op with incompatible scale type}}
  %output, %invstd = "onnx.RMSLayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, tensor<4xf16>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none)
  return %output : tensor<1x2x4xf32>
}
