// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --onnx-hybrid-transform="enable-convtranspose-phased=true" --onnx-form-deferred-foldable-expressions %s | FileCheck %s

// Constant propagation in the hybrid transform must not hide an initializer
// expression before decomposition patterns have consumed its visible QDQ.
func.func @test_deferred_convtrans_stride11(%arg0: tensor<1x1x12x44xf32>) -> tensor<1x1x13x57xf32> {
  %outputScale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %inputScale = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %weightScale = onnx.Constant dense<1.22070313E-4> : tensor<f32>
  %zeroPoint = onnx.Constant dense<2> : tensor<i8>
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 64 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1x1x4x16xi8>
  %biasScale = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %bias = "onnx.InitializerSource"() {kind = "external_data", length = 1 : i64, name = "bias", offset = 64 : i64, path = "/not/read", scope = []} : () -> tensor<1xi8>
  %dequantizedBias = "onnx.DequantizeLinear"(%bias, %biasScale, %zeroPoint) {axis = 1 : si64} : (tensor<1xi8>, tensor<f32>, tensor<i8>) -> tensor<1xf32>
  %dequantizedWeight = "onnx.DequantizeLinear"(%weight, %weightScale, %zeroPoint) {axis = 1 : si64} : (tensor<1x1x4x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x4x16xf32>
  %quantizedInput = "onnx.QuantizeLinear"(%arg0, %inputScale, %zeroPoint) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x12x44xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xi8>
  %dequantizedInput = "onnx.DequantizeLinear"(%quantizedInput, %inputScale, %zeroPoint) {axis = 1 : si64} : (tensor<1x1x12x44xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x12x44xf32>
  %convTranspose = "onnx.ConvTranspose"(%dequantizedInput, %dequantizedWeight, %dequantizedBias) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [4, 16], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x12x44xf32>, tensor<1x1x4x16xf32>, tensor<1xf32>) -> tensor<1x1x13x57xf32>
  %quantizedOutput = "onnx.QuantizeLinear"(%convTranspose, %outputScale, %zeroPoint) {axis = 1 : si64, saturate = 1 : si64} : (tensor<1x1x13x57xf32>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xi8>
  %dequantizedOutput = "onnx.DequantizeLinear"(%quantizedOutput, %outputScale, %zeroPoint) {axis = 1 : si64} : (tensor<1x1x13x57xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x13x57xf32>
  onnx.Return %dequantizedOutput : tensor<1x1x13x57xf32>

// CHECK-LABEL: func.func @test_deferred_convtrans_stride11
// CHECK:         %[[WEIGHT:.+]] = "onnx.InitializerSource"() {{.*}}name = "weight"
// CHECK:         %[[DEFERRED_WEIGHT:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]]
// CHECK-NOT:     "onnx.ConvTranspose"
// CHECK:         "onnx.Conv"({{.*}}, %[[DEFERRED_WEIGHT]], {{.*}}) {{.*}}kernel_shape = [4, 16]
// CHECK:         onnx.Return
}
