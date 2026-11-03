// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s --split-input-file | FileCheck %s

// -----

func.func @mish_f16_swapped(%x: tensor<?x3xf16>) -> tensor<?x3xf16> {
  %softplus = "onnx.Softplus"(%x) : (tensor<?x3xf16>) -> tensor<?x3xf16>
  %tanh = "onnx.Tanh"(%softplus) : (tensor<?x3xf16>) -> tensor<?x3xf16>
  %mish = "onnx.Mul"(%tanh, %x) : (tensor<?x3xf16>, tensor<?x3xf16>) -> tensor<?x3xf16>
  return %mish : tensor<?x3xf16>
}

// CHECK-LABEL: func.func @mish_f16_swapped(
// CHECK-SAME:  %arg0: tensor<?x3xf16>
// CHECK-NEXT:  %0 = "onnx.Mish"(%arg0) : (tensor<?x3xf16>) -> tensor<?x3xf16>
// CHECK-NEXT:  return %0 : tensor<?x3xf16>

// -----

func.func @mish_bf16(%x: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  %softplus = "onnx.Softplus"(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  %tanh = "onnx.Tanh"(%softplus) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  %mish = "onnx.Mul"(%x, %tanh) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %mish : tensor<2x3xbf16>
}

// CHECK-LABEL: func.func @mish_bf16(
// CHECK-SAME:  %arg0: tensor<2x3xbf16>
// CHECK-NEXT:  %0 = "onnx.Mish"(%arg0) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
// CHECK-NEXT:  return %0 : tensor<2x3xbf16>

// -----

func.func @mish_quantized(%x: tensor<2x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x3x!quant.uniform<i8:f32, 0.1>> {
  %softplus = "onnx.Softplus"(%x) : (tensor<2x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x3x!quant.uniform<i8:f32, 0.1>>
  %tanh = "onnx.Tanh"(%softplus) : (tensor<2x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x3x!quant.uniform<i8:f32, 0.1>>
  %mish = "onnx.Mul"(%x, %tanh) : (tensor<2x3x!quant.uniform<i8:f32, 0.1>>, tensor<2x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x3x!quant.uniform<i8:f32, 0.1>>
  return %mish : tensor<2x3x!quant.uniform<i8:f32, 0.1>>
}

// CHECK-LABEL: func.func @mish_quantized(
// CHECK-SAME:  %arg0: tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK-NOT:   "onnx.Mish"
// CHECK:       "onnx.Softplus"
// CHECK-NOT:   "onnx.Mish"
// CHECK:       "onnx.Tanh"
// CHECK-NOT:   "onnx.Mish"
// CHECK:       "onnx.Mul"
// CHECK-NOT:   "onnx.Mish"
// CHECK:       return

// -----

func.func @mish_different_input(%x: tensor<2x3xf16>, %other: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %softplus = "onnx.Softplus"(%other) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %tanh = "onnx.Tanh"(%softplus) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %mish = "onnx.Mul"(%x, %tanh) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %mish : tensor<2x3xf16>
}

// CHECK-LABEL: func.func @mish_different_input(
// CHECK-SAME:  %arg0: tensor<2x3xf16>
// CHECK-NOT:   "onnx.Mish"
// CHECK:       "onnx.Softplus"
// CHECK-NOT:   "onnx.Mish"
// CHECK:       "onnx.Tanh"
// CHECK-NOT:   "onnx.Mish"
// CHECK:       "onnx.Mul"
// CHECK-NOT:   "onnx.Mish"
// CHECK:       return

// -----

func.func @mish_shared_tanh(%x: tensor<2x3xf16>) -> (tensor<2x3xf16>, tensor<2x3xf16>) {
  %softplus = "onnx.Softplus"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %tanh = "onnx.Tanh"(%softplus) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %mul = "onnx.Mul"(%x, %tanh) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %mul, %tanh : tensor<2x3xf16>, tensor<2x3xf16>
}

// CHECK-LABEL: func.func @mish_shared_tanh(
// CHECK-SAME:  %arg0: tensor<2x3xf16>
// CHECK:       %0 = "onnx.Softplus"(%arg0)
// CHECK:       %1 = "onnx.Tanh"(%0)
// CHECK:       %2 = "onnx.Mul"(%arg0, %1)
// CHECK-NOT:   "onnx.Mish"
// CHECK:       return %2, %1

// -----

func.func @mish_shared_softplus(%x: tensor<2x3xf16>) -> (tensor<2x3xf16>, tensor<2x3xf16>) {
  %softplus = "onnx.Softplus"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %tanh = "onnx.Tanh"(%softplus) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %mul = "onnx.Mul"(%x, %tanh) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %mul, %softplus : tensor<2x3xf16>, tensor<2x3xf16>
}

// CHECK-LABEL: func.func @mish_shared_softplus(
// CHECK-SAME:  %arg0: tensor<2x3xf16>
// CHECK:       %0 = "onnx.Softplus"(%arg0)
// CHECK:       %1 = "onnx.Tanh"(%0)
// CHECK:       %2 = "onnx.Mul"(%arg0, %1)
// CHECK-NOT:   "onnx.Mish"
// CHECK:       return %2, %0
