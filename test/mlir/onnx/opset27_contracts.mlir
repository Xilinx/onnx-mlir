// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates.
// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @bitcast
func.func @bitcast(%x: tensor<2x3xf32>) -> tensor<*xi32> {
  // CHECK: "onnx.BitCast"{{.*}} -> tensor<2x3xi32>
  %y = "onnx.BitCast"(%x) {to = i32} : (tensor<2x3xf32>) -> tensor<*xi32>
  return %y : tensor<*xi32>
}

// -----

// CHECK-LABEL: func.func @cumprod
func.func @cumprod(%x: tensor<2x3xf32>, %axis: tensor<i64>) -> tensor<*xf32> {
  // CHECK: "onnx.CumProd"{{.*}} -> tensor<2x3xf32>
  %y = "onnx.CumProd"(%x, %axis) {exclusive = 1 : si64, reverse = 1 : si64}
      : (tensor<2x3xf32>, tensor<i64>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}

// -----

// CHECK-LABEL: func.func @tensor_scatter
func.func @tensor_scatter(%cache: tensor<2x4x8x16xf32>,
    %update: tensor<2x4x1x16xf32>, %indices: tensor<2xi64>) -> tensor<*xf32> {
  // CHECK: "onnx.TensorScatter"{{.*}} -> tensor<2x4x8x16xf32>
  %y = "onnx.TensorScatter"(%cache, %update, %indices)
      {axis = 2 : si64, mode = "circular"}
      : (tensor<2x4x8x16xf32>, tensor<2x4x1x16xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}

// -----

// CHECK-LABEL: func.func @dequantize_output_type
func.func @dequantize_output_type(%x: tensor<2x3xui8>, %scale: tensor<f16>,
    %zero: tensor<ui8>) -> tensor<*xf32> {
  // CHECK: "onnx.DequantizeLinear"{{.*}}output_dtype = 1 : si64{{.*}} -> tensor<2x3xf32>
  %y = "onnx.DequantizeLinear"(%x, %scale, %zero) {output_dtype = 1 : si64}
      : (tensor<2x3xui8>, tensor<f16>, tensor<ui8>) -> tensor<*xf32>
  return %y : tensor<*xf32>
}
