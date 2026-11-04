// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.
// RUN: onnx-mlir-opt --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @quantize_fp32_arithmetic
func.func @quantize_fp32_arithmetic(%x: tensor<2xf16>, %scale: tensor<f16>,
    %zero: tensor<i8>) -> tensor<2xi8> {
  // CHECK-DAG: tosa.cast %arg0 : (tensor<2xf16>) -> tensor<2xf32>
  // CHECK-DAG: tosa.cast %arg1 : (tensor<f16>) -> tensor<f32>
  // CHECK: tosa.mul {{.*}} -> tensor<2xf32>
  // CHECK-NOT: "onnx.QuantizeLinear"
  %y = "onnx.QuantizeLinear"(%x, %scale, %zero) {precision = 1 : si64}
      : (tensor<2xf16>, tensor<f16>, tensor<i8>) -> tensor<2xi8>
  return %y : tensor<2xi8>
}

// -----

// CHECK-LABEL: func.func @dequantize_fp32_arithmetic
func.func @dequantize_fp32_arithmetic(%x: tensor<2xi8>, %scale: tensor<f16>,
    %zero: tensor<i8>) -> tensor<2xf32> {
  // CHECK: tosa.cast %arg1 : (tensor<f16>) -> tensor<f32>
  // CHECK: tosa.mul {{.*}} -> tensor<2xf32>
  // CHECK-NOT: "onnx.DequantizeLinear"
  %y = "onnx.DequantizeLinear"(%x, %scale, %zero) {output_dtype = 1 : si64}
      : (tensor<2xi8>, tensor<f16>, tensor<i8>) -> tensor<2xf32>
  return %y : tensor<2xf32>
}
