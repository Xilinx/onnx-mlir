// RUN: onnx-mlir-opt --add-requant-for-output-conv %s --split-input-file | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

//===----------------------------------------------------------------------===//
// Positive Tests: Should insert an identity XCOMPILERRequantize between the
// multi-use quantized producer and the DequantizeLinear consumer.
//===----------------------------------------------------------------------===//

// Test 1: Conv with two uses (DQ output + another Conv input) should get an
// identity Requantize inserted on the DQ branch only.
// CHECK-LABEL: @conv_multi_use_with_dq
func.func @conv_multi_use_with_dq(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> (tensor<1x4x6x6xf32>, tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>) {
  %scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %w = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %none = "onnx.NoValue"() {value} : () -> none
  %conv = "onnx.Conv"(%arg0, %w, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, none) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %dq = "onnx.DequantizeLinear"(%conv, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>, tensor<f32>, tensor<i8>) -> tensor<1x4x6x6xf32>
  return %dq, %conv : tensor<1x4x6x6xf32>, tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>
}
// CHECK: %[[CONV:.+]] = "onnx.Conv"
// CHECK: %[[RQ:.+]] = "onnx.XCOMPILERRequantize"(%[[CONV]])
// CHECK-SAME: a_scale = [2.500000e-01
// CHECK-SAME: a_zero_point = [0]
// CHECK-SAME: y_scale = [2.500000e-01
// CHECK-SAME: y_zero_point = [0]
// CHECK: %[[DQ:.+]] = "onnx.DequantizeLinear"(%[[RQ]]
// CHECK: return %[[DQ]], %[[CONV]]

// -----

// Test 2: XCOMPILERDepthwiseConv with two uses + DQ -> Requantize inserted.
// CHECK-LABEL: @depthwise_conv_multi_use_with_dq
func.func @depthwise_conv_multi_use_with_dq(
    %arg0: tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>,
    %weight: tensor<1x3x3x16x!quant.uniform<i8:f32, 0.01>>,
    %bias: tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>) -> (tensor<1x8x8x16xf32>, tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>) {
  %scale = onnx.Constant dense<4.000000e-02> : tensor<f32>
  %zp = onnx.Constant dense<-128> : tensor<i8>
  %conv = "onnx.XCOMPILERDepthwiseConv"(%arg0, %weight, %bias) {activation = "NONE", auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.05:128>>, tensor<1x3x3x16x!quant.uniform<i8:f32, 0.01>>, tensor<16x!quant.uniform<i32:f32, 5.000000e-04>>) -> tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>
  %dq = "onnx.DequantizeLinear"(%conv, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>, tensor<f32>, tensor<i8>) -> tensor<1x8x8x16xf32>
  return %dq, %conv : tensor<1x8x8x16xf32>, tensor<1x8x8x16x!quant.uniform<u8:f32, 0.04:128>>
}
// CHECK: %[[CONV:.+]] = "onnx.XCOMPILERDepthwiseConv"
// CHECK: %[[RQ:.+]] = "onnx.XCOMPILERRequantize"(%[[CONV]])
// CHECK-SAME: a_zero_point = [128]
// CHECK-SAME: y_zero_point = [128]
// CHECK: "onnx.DequantizeLinear"(%[[RQ]]

// -----

// Test 3: XCOMPILERFusedEltwise with two uses + DQ -> Requantize inserted.
// CHECK-LABEL: @fused_eltwise_multi_use_with_dq
func.func @fused_eltwise_multi_use_with_dq(
    %arg0: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>,
    %arg1: tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>) -> (tensor<1x16x32x32xf32>, tensor<1x16x32x32x!quant.uniform<u8:f32, 0.10:120>>) {
  %scale = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<120> : tensor<ui8>
  %add = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {nonlinear = "NONE", type = "ADD"} : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>, tensor<1x16x32x32x!quant.uniform<u8:f32, 0.08:128>>) -> tensor<1x16x32x32x!quant.uniform<u8:f32, 0.10:120>>
  %dq = "onnx.DequantizeLinear"(%add, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x16x32x32x!quant.uniform<u8:f32, 0.10:120>>, tensor<f32>, tensor<ui8>) -> tensor<1x16x32x32xf32>
  return %dq, %add : tensor<1x16x32x32xf32>, tensor<1x16x32x32x!quant.uniform<u8:f32, 0.10:120>>
}
// CHECK: %[[ELT:.+]] = "onnx.XCOMPILERFusedEltwise"
// CHECK: %[[RQ:.+]] = "onnx.XCOMPILERRequantize"(%[[ELT]])
// CHECK-SAME: a_scale = [1.000000e-01
// CHECK-SAME: a_zero_point = [120]
// CHECK-SAME: y_scale = [1.000000e-01
// CHECK-SAME: y_zero_point = [120]
// CHECK: "onnx.DequantizeLinear"(%[[RQ]]

// -----

// Test 4: Per-axis quantized Conv with multi-use + DQ -> per-axis Requantize.
// CHECK-LABEL: @conv_per_axis_multi_use_with_dq
func.func @conv_per_axis_multi_use_with_dq(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> (tensor<1x4x6x6xf32>, tensor<1x4x6x6x!quant.uniform<u8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) {
  %scale = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %w = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %none = "onnx.NoValue"() {value} : () -> none
  %conv = "onnx.Conv"(%arg0, %w, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, none) -> tensor<1x4x6x6x!quant.uniform<u8:f32:1, {0.1, 0.2, 0.3, 0.4}>>
  %dq = "onnx.DequantizeLinear"(%conv, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x6x6x!quant.uniform<u8:f32:1, {0.1, 0.2, 0.3, 0.4}>>, tensor<f32>, tensor<i8>) -> tensor<1x4x6x6xf32>
  return %dq, %conv : tensor<1x4x6x6xf32>, tensor<1x4x6x6x!quant.uniform<u8:f32:1, {0.1, 0.2, 0.3, 0.4}>>
}
// CHECK: %[[CONV:.+]] = "onnx.Conv"
// CHECK: %[[RQ:.+]] = "onnx.XCOMPILERRequantize"(%[[CONV]])
// CHECK-SAME: a_scale = [1.000000e-01 : f32, 2.000000e-01 : f32, 3.000000e-01 : f32, 4.000000e-01 : f32]
// CHECK-SAME: a_zero_point = [0, 0, 0, 0]
// CHECK-SAME: y_scale = [1.000000e-01 : f32, 2.000000e-01 : f32, 3.000000e-01 : f32, 4.000000e-01 : f32]
// CHECK-SAME: y_zero_point = [0, 0, 0, 0]
// CHECK: "onnx.DequantizeLinear"(%[[RQ]]

// -----

//===----------------------------------------------------------------------===//
// Negative Tests: Should NOT insert Requantize.
//===----------------------------------------------------------------------===//

// Test 5: Conv with a single use (only DQ) should not be modified - the
// "buffer" is unnecessary when no other consumers exist.
// CHECK-LABEL: @conv_single_use_with_dq
func.func @conv_single_use_with_dq(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x4x6x6xf32> {
  %scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %w = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %none = "onnx.NoValue"() {value} : () -> none
  %conv = "onnx.Conv"(%arg0, %w, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, none) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %dq = "onnx.DequantizeLinear"(%conv, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>, tensor<f32>, tensor<i8>) -> tensor<1x4x6x6xf32>
  return %dq : tensor<1x4x6x6xf32>
}
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: onnx.Conv
// CHECK: onnx.DequantizeLinear

// -----

// Test 6: Non-targeted producer (Relu) feeding DQ should not match even with
// multiple uses; the pass only targets Conv / DepthwiseConv / FusedEltwise.
// CHECK-LABEL: @non_targeted_producer
func.func @non_targeted_producer(%arg0: tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> (tensor<1x4x6x6xf32>, tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>) {
  %scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %relu = "onnx.Relu"(%arg0) : (tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %dq = "onnx.DequantizeLinear"(%relu, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>, tensor<f32>, tensor<i8>) -> tensor<1x4x6x6xf32>
  return %dq, %relu : tensor<1x4x6x6xf32>, tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>
}
// CHECK-NOT: onnx.XCOMPILERRequantize
// CHECK: onnx.Relu
// CHECK: onnx.DequantizeLinear

// -----

// Test 7: Conv producing non-quantized (f32) output - no quant params to
// derive the identity Requantize from.
// CHECK-LABEL: @conv_non_quantized
func.func @conv_non_quantized(%arg0: tensor<1x3x8x8xf32>) -> (tensor<1x4x6x6xf32>, tensor<1x4x6x6xf32>) {
  %w = onnx.Constant {value = dense<1.0> : tensor<4x3x3x3xf32>} : tensor<4x3x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %conv = "onnx.Conv"(%arg0, %w, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, none) -> tensor<1x4x6x6xf32>
  // Two uses of %conv, but no quant types -> pass should leave it alone.
  return %conv, %conv : tensor<1x4x6x6xf32>, tensor<1x4x6x6xf32>
}
// CHECK-NOT: onnx.XCOMPILERRequantize

// -----

// Test 8: Conv with two DQ siblings (still multi-use). Each DQ should get its
// own Requantize inserted so neither remains as a direct consumer of Conv.
// CHECK-LABEL: @conv_with_two_dq_consumers
func.func @conv_with_two_dq_consumers(%arg0: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> (tensor<1x4x6x6xf32>, tensor<1x4x6x6xf32>) {
  %scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %w = onnx.Constant {value = dense<1> : tensor<4x3x3x3xi8>} : tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %none = "onnx.NoValue"() {value} : () -> none
  %conv = "onnx.Conv"(%arg0, %w, %none) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4x3x3x3x!quant.uniform<i8:f32, 1.000000e+00>>, none) -> tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>
  %dq1 = "onnx.DequantizeLinear"(%conv, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>, tensor<f32>, tensor<i8>) -> tensor<1x4x6x6xf32>
  %dq2 = "onnx.DequantizeLinear"(%conv, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4x6x6x!quant.uniform<u8:f32, 2.500000e-01:0>>, tensor<f32>, tensor<i8>) -> tensor<1x4x6x6xf32>
  return %dq1, %dq2 : tensor<1x4x6x6xf32>, tensor<1x4x6x6xf32>
}
// CHECK: %[[CONV:.+]] = "onnx.Conv"
// CHECK: "onnx.XCOMPILERRequantize"(%[[CONV]])
// CHECK: "onnx.XCOMPILERRequantize"(%[[CONV]])
