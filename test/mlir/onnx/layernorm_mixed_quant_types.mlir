// RUN: onnx-mlir-opt %s --split-input-file --verify-diagnostics -o /dev/null

func.func @float_input_quantized_affine(%input: tensor<1x2x4xf32>, %scale: tensor<4x!quant.uniform<i16:f32, 1.0>>, %bias: tensor<4x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x2x4xf32> {
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i16:f32, 1.0>>, tensor<4x!quant.uniform<i16:f32, 1.0>>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @quantized_input_float_affine(%input: tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, %scale: tensor<4xf32>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, tensor<4xf32>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @different_quantization_parameters(%input: tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, %scale: tensor<4x!quant.uniform<i8:f32, 0.5:3>>, %bias: tensor<4x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x2x4xf32> {
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4x!quant.uniform<i16:f32, 0.25>>, tensor<4x!quant.uniform<i8:f32, 0.5:3>>, tensor<4x!quant.uniform<i16:f32, 1.0>>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @per_axis_affine(%input: tensor<1x2x4xf32>, %scale: tensor<4x!quant.uniform<i8:f32:0, {0.25, 0.5, 0.75, 1.0}>>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i8:f32:0, {0.25, 0.5, 0.75, 1.0}>>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @optional_bias(%input: tensor<1x2x4xf32>, %scale: tensor<4x!quant.uniform<i8:f32, 0.5>>) -> tensor<1x2x4xf32> {
  %absent = "onnx.NoValue"() {value} : () -> none
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %absent) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i8:f32, 0.5>>, none) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @rms_float_input_quantized_affine(%input: tensor<1x2x4xf32>, %scale: tensor<4x!quant.uniform<i16:f32, 1.0>>, %bias: tensor<4x!quant.uniform<i16:f32, 1.0>>) -> tensor<1x2x4xf32> {
  %output, %invstd = "onnx.RMSLayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4x!quant.uniform<i16:f32, 1.0>>, tensor<4x!quant.uniform<i16:f32, 1.0>>) -> (tensor<1x2x4xf32>, none)
  return %output : tensor<1x2x4xf32>
}

// -----

func.func @plain_float(%input: tensor<1x2x4xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf32>) -> tensor<1x2x4xf32> {
  %output, %mean, %invstd = "onnx.LayerNormalization"(%input, %scale, %bias) {axis = -1 : si64} : (tensor<1x2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<1x2x4xf32>, none, none)
  return %output : tensor<1x2x4xf32>
}
