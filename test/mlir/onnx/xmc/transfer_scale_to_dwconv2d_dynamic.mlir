// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// RUN: onnx-mlir-opt %s -transfer-scale-to-dwconv2d | FileCheck %s
// RUN: onnx-mlir-opt %s -transfer-scale-to-dwconv2d -shape-inference -o /dev/null

// Static model inputs can still produce runtime intermediate dimensions. Do not
// put the dynamic-dimension sentinel into the input or output Reshape constants.
func.func @runtime_input(%input: tensor<1x?x1024xf32>) -> tensor<1x?x1024xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<1024xf32>} : () -> tensor<1024xf32>
  %scaled = "onnx.Mul"(%input, %scale) : (tensor<1x?x1024xf32>, tensor<1024xf32>) -> tensor<1x?x1024xf32>
  return %scaled : tensor<1x?x1024xf32>
}
// CHECK-LABEL: func.func @runtime_input
// CHECK: onnx.Constant
// CHECK-NEXT: {{.*}}onnx.Mul
// CHECK-NEXT: return

// Four-dimensional inputs also require a static output Reshape target.
func.func @runtime_4d_input(%input: tensor<?x2x?x16xf32>) -> tensor<?x2x?x16xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %scaled = "onnx.Mul"(%scale, %input) : (tensor<16xf32>, tensor<?x2x?x16xf32>) -> tensor<?x2x?x16xf32>
  return %scaled : tensor<?x2x?x16xf32>
}
// CHECK-LABEL: func.func @runtime_4d_input
// CHECK: onnx.Constant
// CHECK-NEXT: {{.*}}onnx.Mul
// CHECK-NEXT: return

// The input can be static before the result type has been refined.
func.func @runtime_mul_output(%input: tensor<1x2x16xf32>) -> tensor<1x?x16xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %scaled = "onnx.Mul"(%input, %scale) : (tensor<1x2x16xf32>, tensor<16xf32>) -> tensor<1x?x16xf32>
  return %scaled : tensor<1x?x16xf32>
}
// CHECK-LABEL: func.func @runtime_mul_output
// CHECK: onnx.Constant
// CHECK-NEXT: {{.*}}onnx.Mul
// CHECK-NEXT: return

// Check the following activation's result before creating any operations.
func.func @runtime_activation_output(%input: tensor<1x2x16xf32>) -> tensor<1x?x16xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %scaled = "onnx.Mul"(%input, %scale) : (tensor<1x2x16xf32>, tensor<16xf32>) -> tensor<1x2x16xf32>
  %activated = "onnx.Relu"(%scaled) : (tensor<1x2x16xf32>) -> tensor<1x?x16xf32>
  return %activated : tensor<1x?x16xf32>
}
// CHECK-LABEL: func.func @runtime_activation_output
// CHECK: onnx.Constant
// CHECK-NEXT: {{.*}}onnx.Mul
// CHECK-NEXT: {{.*}}onnx.Relu
// CHECK-NEXT: return

func.func @unranked_mul_output(%input: tensor<1x2x16xf32>) -> tensor<*xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %scaled = "onnx.Mul"(%input, %scale) : (tensor<1x2x16xf32>, tensor<16xf32>) -> tensor<*xf32>
  return %scaled : tensor<*xf32>
}
// CHECK-LABEL: func.func @unranked_mul_output
// CHECK: onnx.Constant
// CHECK-NEXT: {{.*}}onnx.Mul
// CHECK-NEXT: return

func.func @unranked_activation_output(%input: tensor<1x2x16xf32>) -> tensor<*xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %scaled = "onnx.Mul"(%input, %scale) : (tensor<1x2x16xf32>, tensor<16xf32>) -> tensor<1x2x16xf32>
  %activated = "onnx.Relu"(%scaled) : (tensor<1x2x16xf32>) -> tensor<*xf32>
  return %activated : tensor<*xf32>
}
// CHECK-LABEL: func.func @unranked_activation_output
// CHECK: onnx.Constant
// CHECK-NEXT: {{.*}}onnx.Mul
// CHECK-NEXT: {{.*}}onnx.Relu
// CHECK-NEXT: return

// Keep converting the original static scale + activation case.
func.func @static_activation(%input: tensor<1x2x16xf32>) -> tensor<1x2x16xf32> {
  %scale = "onnx.Constant"() {value = dense<2.0> : tensor<16xf32>} : () -> tensor<16xf32>
  %scaled = "onnx.Mul"(%input, %scale) : (tensor<1x2x16xf32>, tensor<16xf32>) -> tensor<1x2x16xf32>
  %activated = "onnx.Relu"(%scaled) : (tensor<1x2x16xf32>) -> tensor<1x2x16xf32>
  return %activated : tensor<1x2x16xf32>
}
// CHECK-LABEL: func.func @static_activation
// CHECK: onnx.Reshape{{.*}} -> tensor<1x1x2x16xf32>
// CHECK: onnx.XFEConv{{.*}}group = 16
// CHECK: onnx.Relu{{.*}} -> tensor<1x1x2x16xf32>
// CHECK: onnx.Reshape{{.*}} -> tensor<1x2x16xf32>
// CHECK: return
