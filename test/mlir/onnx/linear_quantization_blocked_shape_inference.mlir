// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @quantize_blocked_1d
// CHECK-SAME: -> tensor<4xui8>
func.func @quantize_blocked_1d(%x: tensor<4xf32>, %scale: tensor<2xf32>, %zp: tensor<2xui8>) -> tensor<*xui8> {
  %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xf32>, tensor<2xf32>, tensor<2xui8>) -> tensor<*xui8>
  return %y : tensor<*xui8>
}

// -----

// CHECK-LABEL: func.func @dequantize_blocked_1d
// CHECK-SAME: -> tensor<4xf32>
func.func @dequantize_blocked_1d(%x: tensor<4xui8>, %scale: tensor<2xf32>, %zp: tensor<2xui8>) -> tensor<*xf32> {
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}

// -----

// CHECK-LABEL: func.func @quantize_blocked_dynamic
// CHECK-SAME: -> tensor<?xui8>
func.func @quantize_blocked_dynamic(%x: tensor<?xf32>, %scale: tensor<2xf32>) -> tensor<*xui8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 0 : si64, block_size = 2 : si64} : (tensor<?xf32>, tensor<2xf32>, none) -> tensor<*xui8>
  return %y : tensor<*xui8>
}

// -----

// CHECK-LABEL: func.func @dequantize_blocked_dynamic
// CHECK-SAME: -> tensor<?xf32>
func.func @dequantize_blocked_dynamic(%x: tensor<?xui8>, %scale: tensor<2xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 0 : si64, block_size = 2 : si64} : (tensor<?xui8>, tensor<2xf32>, none) -> tensor<*xf32>
  return %y : tensor<*xf32>
}

// -----

// CHECK-LABEL: func.func @dequantize_dynamic_scalar_block_size
// CHECK-SAME: -> tensor<?x7xf32>
func.func @dequantize_dynamic_scalar_block_size(%x: tensor<?x7xui8>, %scale: tensor<?xf32>, %zp: tensor<ui8>) -> tensor<*xf32> {
  %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 2 : si64} : (tensor<?x7xui8>, tensor<?xf32>, tensor<ui8>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}
