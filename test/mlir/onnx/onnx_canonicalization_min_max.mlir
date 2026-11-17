// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s

func.func @max_1_2x3xf32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_1_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[A0]]

// -----

func.func @max_1_2x3xi32(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Max"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @max_1_2x3xi32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[A0]]

// -----

func.func @max_1_dynamicx3xf32(%arg0: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Max"(%arg0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @max_1_dynamicx3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[A0]]

// -----

func.func @max_1_unrankedxf32(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @max_1_unrankedxf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<*xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[A0]]

// -----

func.func @max_1_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Max"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @max_1_f32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<f32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[A0]]

// -----

func.func @max_2_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_2_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]])
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @max_3_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_3_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]], [[A2]])
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @min_1_2x3xf32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_1_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[A0]]

// -----

func.func @min_1_2x3xi32(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Min"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @min_1_2x3xi32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[A0]]

// -----

func.func @min_1_dynamicx3xf32(%arg0: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Min"(%arg0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @min_1_dynamicx3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[A0]]

// -----

func.func @min_1_unrankedxf32(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @min_1_unrankedxf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<*xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[A0]]

// -----

func.func @min_1_f32(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Min"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @min_1_f32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<f32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[A0]]

// -----

func.func @min_2_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_2_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]])
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R]]

// -----

func.func @min_3_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_3_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]], [[A2]])
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R]]

// -----

