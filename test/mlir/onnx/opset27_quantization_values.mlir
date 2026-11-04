// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.
// RUN: onnx-mlir-opt --split-input-file %s -constprop-onnx="enable-quant-const-fold enable-dequant-const-fold" | FileCheck %s

// Precision is observable with existing types: this value rounds to 0.5 in
// FP16, so the quotient rounds to even zero before quantization.
func.func @precision_fp16() -> tensor<ui8> {
  %x = onnx.Constant dense<0.5001220703125> : tensor<f32>
  %s = onnx.Constant dense<1.0> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %y = "onnx.QuantizeLinear"(%x, %s, %z) {precision = 10 : si64}
      : (tensor<f32>, tensor<f32>, tensor<ui8>) -> tensor<ui8>
  return %y : tensor<ui8>
}
// CHECK-LABEL: @precision_fp16
// CHECK: onnx.Constant dense<0> : tensor<ui8>
// CHECK-NOT: onnx.QuantizeLinear

// -----

// In FP32 the same quotient is greater than 0.5 and rounds to one.
func.func @precision_fp32() -> tensor<ui8> {
  %x = onnx.Constant dense<0.5001220703125> : tensor<f32>
  %s = onnx.Constant dense<1.0> : tensor<f32>
  %z = onnx.Constant dense<0> : tensor<ui8>
  %y = "onnx.QuantizeLinear"(%x, %s, %z) {precision = 1 : si64}
      : (tensor<f32>, tensor<f32>, tensor<ui8>) -> tensor<ui8>
  return %y : tensor<ui8>
}
// CHECK-LABEL: @precision_fp32
// CHECK: onnx.Constant dense<1> : tensor<ui8>
// CHECK-NOT: onnx.QuantizeLinear

// -----

func.func @round_before_zero_point() -> tensor<4xui8> {
  %x = onnx.Constant dense<[0.5, 1.5, 2.5, 3.5]> : tensor<4xf32>
  %s = onnx.Constant dense<1.0> : tensor<f32>
  %z = onnx.Constant dense<1> : tensor<ui8>
  %y = "onnx.QuantizeLinear"(%x, %s, %z)
      : (tensor<4xf32>, tensor<f32>, tensor<ui8>) -> tensor<4xui8>
  return %y : tensor<4xui8>
}
// CHECK-LABEL: @round_before_zero_point
// CHECK: onnx.Constant dense<[1, 3, 3, 5]> : tensor<4xui8>
// CHECK-NOT: onnx.QuantizeLinear

// -----

func.func @dequantize_fp32_output() -> tensor<2xf32> {
  %x = onnx.Constant dense<[3, 5]> : tensor<2xui8>
  %s = onnx.Constant dense<0.5> : tensor<f16>
  %z = onnx.Constant dense<1> : tensor<ui8>
  %y = "onnx.DequantizeLinear"(%x, %s, %z) {output_dtype = 1 : si64}
      : (tensor<2xui8>, tensor<f16>, tensor<ui8>) -> tensor<2xf32>
  return %y : tensor<2xf32>
}
// CHECK-LABEL: @dequantize_fp32_output
// CHECK: onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK-NOT: onnx.DequantizeLinear
