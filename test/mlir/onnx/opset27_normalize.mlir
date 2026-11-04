// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.
// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @quantization_precision
func.func @quantization_precision(%x: tensor<2xf16>, %s: tensor<f16>,
    %zp: tensor<ui8>) -> tensor<2xui8> {
  // CHECK: [[X:%.+]] = "onnx.Cast"(%arg0) {{.*}}to = f32
  // CHECK: [[S:%.+]] = "onnx.Cast"(%arg1) {{.*}}to = f32
  // CHECK: "onnx.QuantizeLinear"([[X]], [[S]], %arg2) {{.*}}precision = 0 : si64
  %y = "onnx.QuantizeLinear"(%x, %s, %zp) {precision = 1 : si64}
      : (tensor<2xf16>, tensor<f16>, tensor<ui8>) -> tensor<2xui8>
  return %y : tensor<2xui8>
}

// -----

// CHECK-LABEL: func.func @dequantization_precision
func.func @dequantization_precision(%x: tensor<2xui8>, %s: tensor<f16>,
    %zp: tensor<ui8>) -> tensor<2xf32> {
  // CHECK: [[S:%.+]] = "onnx.Cast"(%arg1) {{.*}}to = f32
  // CHECK: "onnx.DequantizeLinear"(%arg0, [[S]], %arg2) {{.*}}output_dtype = 0 : si64
  %y = "onnx.DequantizeLinear"(%x, %s, %zp) {output_dtype = 1 : si64}
      : (tensor<2xui8>, tensor<f16>, tensor<ui8>) -> tensor<2xf32>
  return %y : tensor<2xf32>
}
