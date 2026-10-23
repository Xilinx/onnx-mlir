// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

func.func @QuantizeLinear_scalar_vector(%x: tensor<2x3xf32>, %scale: tensor<f32>, %zp: tensor<3xi8>) -> tensor<2x3xi8> {

  // expected-error @+1 {{y_zero_point must have the same shape as y_scale}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<f32>, tensor<3xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_vector_scalar(%x: tensor<2x3xf32>, %scale: tensor<3xf32>, %zp: tensor<i8>) -> tensor<2x3xi8> {

  // expected-error @+1 {{y_zero_point must have the same shape as y_scale}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<3xf32>, tensor<i8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_one_vector(%x: tensor<2x3xf32>, %scale: tensor<1xf32>, %zp: tensor<3xi8>) -> tensor<2x3xi8> {

  // expected-error @+1 {{y_zero_point must have the same shape as y_scale}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<1xf32>, tensor<3xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_scalar_pair(%x: tensor<2x3xf32>, %scale: tensor<1xf32>, %zp: tensor<i8>) -> tensor<2x3xi8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<1xf32>, tensor<i8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_vector_pair(%x: tensor<2x3xf32>, %scale: tensor<3xf32>, %zp: tensor<3xi8>) -> tensor<2x3xi8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_unranked(%x: tensor<2x3xf32>, %scale: tensor<*xf32>, %zp: tensor<3xi8>) -> tensor<2x3xi8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<*xf32>, tensor<3xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @DequantizeLinear_scalar_vector(%x: tensor<2x3xi8>, %scale: tensor<f32>, %zp: tensor<3xi8>) -> tensor<2x3xf32> {

  // expected-error @+1 {{x_zero_point must have the same shape as x_scale}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<f32>, tensor<3xi8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_vector_scalar(%x: tensor<2x3xi8>, %scale: tensor<3xf32>, %zp: tensor<i8>) -> tensor<2x3xf32> {

  // expected-error @+1 {{x_zero_point must have the same shape as x_scale}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<3xf32>, tensor<i8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_one_vector(%x: tensor<2x3xi8>, %scale: tensor<1xf32>, %zp: tensor<3xi8>) -> tensor<2x3xf32> {

  // expected-error @+1 {{x_zero_point must have the same shape as x_scale}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<1xf32>, tensor<3xi8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_scalar_pair(%x: tensor<2x3xi8>, %scale: tensor<1xf32>, %zp: tensor<i8>) -> tensor<2x3xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<1xf32>, tensor<i8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_vector_pair(%x: tensor<2x3xi8>, %scale: tensor<3xf32>, %zp: tensor<3xi8>) -> tensor<2x3xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<3xf32>, tensor<3xi8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_unranked(%x: tensor<2x3xi8>, %scale: tensor<*xf32>, %zp: tensor<3xi8>) -> tensor<2x3xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<*xf32>, tensor<3xi8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @QuantizeLinear_axis(%x: tensor<2x3xf32>, %scale: tensor<3xf32>) -> tensor<2x3xi8> {
  %none = "onnx.NoValue"() {value} : () -> none

  // expected-error @+1 {{'axis' value is 2, accepted range is [-2, 1]}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 2 : si64} : (tensor<2x3xf32>, tensor<3xf32>, none) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_length(%x: tensor<2x3xf32>, %scale: tensor<4xf32>) -> tensor<2x3xi8> {
  %none = "onnx.NoValue"() {value} : () -> none

  // expected-error @+1 {{y_scale and y_zero_point 1-D tensor length must match the input axis dim size}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<4xf32>, none) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_rank(%x: tensor<2x3xf32>, %scale: tensor<1x2x3xf32>) -> tensor<2x3xi8> {
  %none = "onnx.NoValue"() {value} : () -> none

  // expected-error @+1 {{y_scale rank needs to be 0, 1 or match x rank}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<1x2x3xf32>, none) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_negative_axis(%x: tensor<2x3xf32>, %scale: tensor<3xf32>) -> tensor<2x3xi8> {
  %none = "onnx.NoValue"() {value} : () -> none

  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<3xf32>, none) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_blocked_1d(%x: tensor<4xf32>, %scale: tensor<2xf32>, %zp: tensor<2xui8>) -> tensor<4xui8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xf32>, tensor<2xf32>, tensor<2xui8>) -> tensor<4xui8>
  return %y : tensor<4xui8>
}

// -----

func.func @QuantizeLinear_dynamic_scalar(%x: tensor<2x3xf32>, %scale: tensor<?xf32>, %zp: tensor<i8>) -> tensor<2x3xi8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<?xf32>, tensor<i8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_scalar_dynamic(%x: tensor<2x3xf32>, %scale: tensor<1xf32>, %zp: tensor<?xi8>) -> tensor<2x3xi8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<1xf32>, tensor<?xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_dynamic_vector(%x: tensor<2x3xf32>, %scale: tensor<?xf32>, %zp: tensor<3xi8>) -> tensor<2x3xi8> {

  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<?xf32>, tensor<3xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_dynamic_vector_length(%x: tensor<2x3xf32>, %scale: tensor<?xf32>, %zp: tensor<4xi8>) -> tensor<2x3xi8> {

  // expected-error @+1 {{y_scale and y_zero_point 1-D tensor length must match the input axis dim size}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<?xf32>, tensor<4xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @QuantizeLinear_dynamic_rank(%x: tensor<2x3xf32>, %scale: tensor<?xf32>, %zp: tensor<2x3xi8>) -> tensor<2x3xi8> {

  // expected-error @+1 {{y_zero_point must have the same shape as y_scale}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<?xf32>, tensor<2x3xi8>) -> tensor<2x3xi8>
  return %y : tensor<2x3xi8>
}

// -----

func.func @DequantizeLinear_dynamic_vector(%x: tensor<2x3xi8>, %scale: tensor<?xf32>, %zp: tensor<3xi8>) -> tensor<2x3xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<?xf32>, tensor<3xi8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_dynamic_scalar(%x: tensor<2x3xi8>, %scale: tensor<?xf32>, %zp: tensor<i8>) -> tensor<2x3xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<?xf32>, tensor<i8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_scalar_dynamic(%x: tensor<2x3xi8>, %scale: tensor<1xf32>, %zp: tensor<?xi8>) -> tensor<2x3xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<1xf32>, tensor<?xi8>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_axis_without_zero_point(%x: tensor<2x3xi8>, %scale: tensor<3xf32>) -> tensor<2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none

  // expected-error @+1 {{'axis' value is 2, accepted range is [-2, 1]}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 2 : si64} : (tensor<2x3xi8>, tensor<3xf32>, none) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_length_without_zero_point(%x: tensor<2x3xi8>, %scale: tensor<4xf32>) -> tensor<2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none

  // expected-error @+1 {{x_scale and x_zero_point 1-D tensor length must match the input axis dim size}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64} : (tensor<2x3xi8>, tensor<4xf32>, none) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_negative_axis_without_zero_point(%x: tensor<2x3xi8>, %scale: tensor<3xf32>) -> tensor<2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none

  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = -1 : si64} : (tensor<2x3xi8>, tensor<3xf32>, none) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @DequantizeLinear_blocked_1d(%x: tensor<4xui8>, %scale: tensor<2xf32>, %zp: tensor<2xui8>) -> tensor<4xf32> {

  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<4xf32>
  return %y : tensor<4xf32>
}

// -----

func.func @QuantizeLinear_blocked_parameter_mismatch(%x: tensor<4xf32>, %scale: tensor<2xf32>, %zp: tensor<3xui8>) -> tensor<4xui8> {
  // expected-error @+1 {{y_zero_point must have the same shape as y_scale}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xf32>, tensor<2xf32>, tensor<3xui8>) -> tensor<4xui8>
  return %y : tensor<4xui8>
}

// -----

func.func @DequantizeLinear_blocked_parameter_mismatch(%x: tensor<4xui8>, %scale: tensor<2xf32>, %zp: tensor<3xui8>) -> tensor<4xf32> {
  // expected-error @+1 {{x_zero_point must have the same shape as x_scale}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xui8>, tensor<2xf32>, tensor<3xui8>) -> tensor<4xf32>
  return %y : tensor<4xf32>
}

// -----

func.func @QuantizeLinear_block_size_one_valid(%x: tensor<5x7xf32>, %scale: tensor<5x7xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 1 : si64} : (tensor<5x7xf32>, tensor<5x7xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_size_negative(%x: tensor<5x7xf32>, %scale: tensor<5x4xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{block_size must be positive for blocked quantization}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = -2 : si64} : (tensor<5x7xf32>, tensor<5x4xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_size_missing(%x: tensor<5x7xf32>, %scale: tensor<5x4xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{block_size must be positive for blocked quantization}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 0 : si64} : (tensor<5x7xf32>, tensor<5x4xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_axis_length(%x: tensor<5x7xf32>, %scale: tensor<5x3xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{y_scale axis dim size must equal ceil(input axis dim size / block_size)}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xf32>, tensor<5x3xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_other_dim(%x: tensor<5x7xf32>, %scale: tensor<4x4xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{y_scale dim 0 must match the input dim size outside the blocking axis}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xf32>, tensor<4x4xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_rank(%x: tensor<5x7xf32>, %scale: tensor<4xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{y_scale rank must match x rank for blocked quantization}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xf32>, tensor<4xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_axis(%x: tensor<5x7xf32>, %scale: tensor<5x4xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{'axis' value is 2, accepted range is [-2, 1]}}
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 2 : si64, block_size = 2 : si64} : (tensor<5x7xf32>, tensor<5x4xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_ceil_valid(%x: tensor<5x7xf32>, %scale: tensor<5x4xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = -1 : si64, block_size = 2 : si64} : (tensor<5x7xf32>, tensor<5x4xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @QuantizeLinear_block_dynamic_valid(%x: tensor<?x7xf32>, %scale: tensor<5x?xf32>) -> tensor<?x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<?x7xf32>, tensor<5x?xf32>, none) -> tensor<?x7xui8>
  return %y : tensor<?x7xui8>
}

// -----

func.func @DequantizeLinear_block_size_one_valid(%x: tensor<5x7xui8>, %scale: tensor<5x7xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 1 : si64} : (tensor<5x7xui8>, tensor<5x7xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_size_negative(%x: tensor<5x7xui8>, %scale: tensor<5x4xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{block_size must be positive for blocked quantization}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = -2 : si64} : (tensor<5x7xui8>, tensor<5x4xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_size_missing(%x: tensor<5x7xui8>, %scale: tensor<5x4xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{block_size must be positive for blocked quantization}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 0 : si64} : (tensor<5x7xui8>, tensor<5x4xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_axis_length(%x: tensor<5x7xui8>, %scale: tensor<5x3xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{x_scale axis dim size must equal ceil(input axis dim size / block_size)}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xui8>, tensor<5x3xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_other_dim(%x: tensor<5x7xui8>, %scale: tensor<4x4xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{x_scale dim 0 must match the input dim size outside the blocking axis}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xui8>, tensor<4x4xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_rank(%x: tensor<5x7xui8>, %scale: tensor<4xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{x_scale rank must match x rank for blocked quantization}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xui8>, tensor<4xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_axis(%x: tensor<5x7xui8>, %scale: tensor<5x4xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{'axis' value is 2, accepted range is [-2, 1]}}
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 2 : si64, block_size = 2 : si64} : (tensor<5x7xui8>, tensor<5x4xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_ceil_valid(%x: tensor<5x7xui8>, %scale: tensor<5x4xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = -1 : si64, block_size = 2 : si64} : (tensor<5x7xui8>, tensor<5x4xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @DequantizeLinear_block_dynamic_valid(%x: tensor<?x7xui8>, %scale: tensor<5x?xf32>) -> tensor<?x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 2 : si64} : (tensor<?x7xui8>, tensor<5x?xf32>, none) -> tensor<?x7xf32>
  return %y : tensor<?x7xf32>
}

// -----

func.func @QuantizeLinear_scalar_with_block_size_valid(%x: tensor<5x7xf32>, %scale: tensor<f32>, %zp: tensor<ui8>) -> tensor<5x7xui8> {
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xf32>, tensor<f32>, tensor<ui8>) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @DequantizeLinear_scalar_with_block_size_valid(%x: tensor<5x7xui8>, %scale: tensor<f32>, %zp: tensor<ui8>) -> tensor<5x7xf32> {
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 2 : si64} : (tensor<5x7xui8>, tensor<f32>, tensor<ui8>) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @QuantizeLinear_single_element_rank2_valid(%x: tensor<5x7xf32>, %scale: tensor<1x1xf32>) -> tensor<5x7xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64} : (tensor<5x7xf32>, tensor<1x1xf32>, none) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @DequantizeLinear_single_element_rank2_valid(%x: tensor<5x7xui8>, %scale: tensor<1x1xf32>) -> tensor<5x7xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64} : (tensor<5x7xui8>, tensor<1x1xf32>, none) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}

// -----

func.func @QuantizeLinear_single_element_rank2_scalar_zero_point_valid(%x: tensor<5x7xf32>, %scale: tensor<1x1xf32>, %zp: tensor<ui8>) -> tensor<5x7xui8> {
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<5x7xf32>, tensor<1x1xf32>, tensor<ui8>) -> tensor<5x7xui8>
  return %y : tensor<5x7xui8>
}

// -----

func.func @DequantizeLinear_single_element_rank2_scalar_zero_point_valid(%x: tensor<5x7xui8>, %scale: tensor<1x1xf32>, %zp: tensor<ui8>) -> tensor<5x7xf32> {
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<5x7xui8>, tensor<1x1xf32>, tensor<ui8>) -> tensor<5x7xf32>
  return %y : tensor<5x7xf32>
}
