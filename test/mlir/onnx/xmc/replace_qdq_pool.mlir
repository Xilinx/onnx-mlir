// RUN: onnx-mlir-opt --replace-qdq-pool %s | FileCheck %s
// NOTE: This pass assumes quant-types has already run, so pool ops use native
// `!quant.uniform` types (no explicit Q/DQ ops).

// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Positive Tests - Redundant DPU coefficient Mul should be removed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_avgpool_3x3_remove_mul
// AvgPool(quantized, 3x3) -> Mul(DPU_coeff=0.984375) -> consumer
// DPU coefficient for 3x3: (3*3*7) / 2^6 = 63/64 = 0.984375
// Mul is redundant and should be removed.
func.func @test_avgpool_3x3_remove_mul(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[POOL]])
  // CHECK: return %[[RELU]]
  // CHECK-NOT: onnx.Mul
}

// -----

// CHECK-LABEL: func.func @test_avgpool_5x5_remove_mul
// AvgPool(quantized, 5x5) -> Mul(DPU_coeff=0.9765625) -> consumer
// DPU coefficient for 5x5: (5*5*10) / 2^8 = 250/256 = 0.9765625
func.func @test_avgpool_5x5_remove_mul(
    %arg0: tensor<1x8x28x28x!quant.uniform<i8:f32, 0.05:0>>)
    -> tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [5, 5],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x8x28x28x!quant.uniform<i8:f32, 0.05:0>>)
    -> tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>>

  %coeff = onnx.Constant dense<9.765625e-01> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>>, tensor<f32>)
      -> tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>>)
      -> tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>>

  return %relu : tensor<1x8x24x24x!quant.uniform<i8:f32, 0.05:0>>

  // CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[POOL]])
  // CHECK: return %[[RELU]]
  // CHECK-NOT: onnx.Mul
}

// -----

// CHECK-LABEL: func.func @test_avgpool_3x3_mul_const_first
// Mul is commutative: constant as first operand should also be removed.
// Mul(DPU_coeff, AvgPool) -> consumer
func.func @test_avgpool_3x3_mul_const_first(
    %arg0: tensor<1x4x16x16x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x4x16x16x!quant.uniform<u8:f32, 0.1:128>>)
    -> tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%coeff, %pool) :
      (tensor<f32>, tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>>)
      -> tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>>

  return %relu : tensor<1x4x14x14x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[POOL]])
  // CHECK: return %[[RELU]]
  // CHECK-NOT: onnx.Mul
}

// -----

// CHECK-LABEL: func.func @test_avgpool_3x3_with_pads
// AvgPool with non-negative padding should still match.
func.func @test_avgpool_3x3_with_pads(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [1, 1, 1, 1],
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: %[[POOL:.*]] = "onnx.AveragePool"(%arg0)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[POOL]])
  // CHECK: return %[[RELU]]
  // CHECK-NOT: onnx.Mul
}

// -----

//===----------------------------------------------------------------------===//
// Negative Tests - Should NOT be transformed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @neg_wrong_coefficient
// Mul constant does not match DPU coefficient for 3x3 kernel -> no transform
func.func @neg_wrong_coefficient(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %wrong = onnx.Constant dense<2.000000e+00> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %wrong) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.AveragePool"
  // CHECK: "onnx.Mul"
  // CHECK: "onnx.Relu"
}

// -----

// CHECK-LABEL: func.func @neg_not_quantized
// AvgPool with float (non-quantized) types -> no transform
func.func @neg_not_quantized(
    %arg0: tensor<1x16x32x32xf32>)
    -> tensor<1x16x30x30xf32> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x16x32x32xf32>) -> tensor<1x16x30x30xf32>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x16x30x30xf32>, tensor<f32>) -> tensor<1x16x30x30xf32>

  return %mul : tensor<1x16x30x30xf32>

  // CHECK: "onnx.AveragePool"
  // CHECK: "onnx.Mul"
}

// -----

// CHECK-LABEL: func.func @neg_avgpool_multiple_uses
// AvgPool result used by both Mul and another consumer -> no transform
func.func @neg_avgpool_multiple_uses(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>,
        tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>) {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu, %pool :
      tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>,
      tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.AveragePool"
  // CHECK: "onnx.Mul"
  // CHECK: "onnx.Relu"
}

// -----

// CHECK-LABEL: func.func @neg_mul_multiple_uses
// Mul result has multiple consumers -> no transform
func.func @neg_mul_multiple_uses(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>,
        tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>) {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu, %mul :
      tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>,
      tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.AveragePool"
  // CHECK: "onnx.Mul"
  // CHECK: "onnx.Relu"
}

// -----

// CHECK-LABEL: func.func @neg_maxpool_not_avgpool
// MaxPool is not AvgPool -> Mul should not be removed
func.func @neg_maxpool_not_avgpool(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>> {

  %pool = "onnx.MaxPoolSingleOut"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      dilations = [1, 1],
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      storage_order = 0 : si64,
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%pool, %coeff) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.MaxPoolSingleOut"
  // CHECK: "onnx.Mul"
  // CHECK: "onnx.Relu"
}

// -----

// CHECK-LABEL: func.func @neg_mul_not_constant
// Mul operand is not a constant -> no transform
func.func @neg_mul_not_constant(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
    %arg1: tensor<f32>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>> {

  %pool = "onnx.AveragePool"(%arg0) {
      auto_pad = "NOTSET",
      ceil_mode = 0 : si64,
      count_include_pad = 0 : si64,
      kernel_shape = [3, 3],
      pads = [0, 0, 0, 0],
      strides = [1, 1]
  } : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %mul = "onnx.Mul"(%pool, %arg1) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.AveragePool"
  // CHECK: "onnx.Mul"
  // CHECK: "onnx.Relu"
}

// -----

// CHECK-LABEL: func.func @neg_no_avgpool_input
// Mul with two non-pool operands -> no transform
func.func @neg_no_avgpool_input(
    %arg0: tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
    -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>> {

  %coeff = onnx.Constant dense<0.984375> : tensor<f32>

  %mul = "onnx.Mul"(%arg0, %coeff) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>, tensor<f32>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  %relu = "onnx.Relu"(%mul) :
      (tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>)
      -> tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  return %relu : tensor<1x16x30x30x!quant.uniform<u8:f32, 0.08:128>>

  // CHECK: "onnx.Mul"
  // CHECK: "onnx.Relu"
}
