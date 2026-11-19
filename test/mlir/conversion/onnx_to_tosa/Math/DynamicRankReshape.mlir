// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=shape. --implicit-check-not=arith.index_cast --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --implicit-check-not=shape. --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=shape. --implicit-check-not='"onnx.Reshape"'

// Keep dynamic computations in ONNX instead of creating auxiliary reshapes.
func.func @mul_dynamic(%lhs: tensor<1x?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<1x?x?xf32> {
  %0 = "onnx.Mul"(%lhs, %rhs) : (tensor<1x?x?xf32>, tensor<?x?xf32>) -> tensor<1x?x?xf32>
  return %0 : tensor<1x?x?xf32>
}

// CHECK-LABEL: func.func @mul_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Mul"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @mul_empty_dynamic(%lhs: tensor<1x0x?xf32>, %rhs: tensor<0x?xf32>) -> tensor<1x0x?xf32> {
  %0 = "onnx.Mul"(%lhs, %rhs) : (tensor<1x0x?xf32>, tensor<0x?xf32>) -> tensor<1x0x?xf32>
  return %0 : tensor<1x0x?xf32>
}

// CHECK-LABEL: func.func @mul_empty_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Mul"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @gemm_dynamic_result(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @gemm_dynamic_result(
// CHECK: %[[RESULT:.*]] = "onnx.Gemm"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @gemm_refined_empty_result(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>) -> tensor<0x?xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<0x?xf32>
  return %0 : tensor<0x?xf32>
}

// CHECK-LABEL: func.func @gemm_refined_empty_result(
// CHECK: %[[RESULT:.*]] = "onnx.Gemm"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @reduce_sum_dynamic(%input: tensor<?x4x?xf32>) -> tensor<?x?xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.ReduceSum"(%input, %axes) {keepdims = 0 : si64} : (tensor<?x4x?xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @reduce_sum_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.ReduceSum"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @add_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Add"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @add_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Add"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @sub_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Sub"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @sub_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Sub"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @div_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Div"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @div_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Div"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @pow_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Pow"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @pow_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Pow"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @equal_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xi1> {
  %0 = "onnx.Equal"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xi1>
  return %0 : tensor<1x?x3xi1>
}

// CHECK-LABEL: func.func @equal_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Equal"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @greater_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xi1> {
  %0 = "onnx.Greater"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xi1>
  return %0 : tensor<1x?x3xi1>
}

// CHECK-LABEL: func.func @greater_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Greater"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @greaterorequal_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xi1> {
  %0 = "onnx.GreaterOrEqual"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xi1>
  return %0 : tensor<1x?x3xi1>
}

// CHECK-LABEL: func.func @greaterorequal_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.GreaterOrEqual"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @less_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xi1> {
  %0 = "onnx.Less"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xi1>
  return %0 : tensor<1x?x3xi1>
}

// CHECK-LABEL: func.func @less_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Less"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @lessorequal_dynamic(%lhs: tensor<1x?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<1x?x3xi1> {
  %0 = "onnx.LessOrEqual"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xi1>
  return %0 : tensor<1x?x3xi1>
}

// CHECK-LABEL: func.func @lessorequal_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.LessOrEqual"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @where_dynamic(%condition: tensor<1x?x3xi1>, %x: tensor<?x3xf32>, %y: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Where"(%condition, %x, %y) : (tensor<1x?x3xi1>, tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @where_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Where"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @prelu_dynamic(%x: tensor<1x?x3xf32>, %slope: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.PRelu"(%x, %slope) : (tensor<1x?x3xf32>, tensor<?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @prelu_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.PRelu"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @batch_norm_dynamic(%x: tensor<1x3x?xf32>, %scale: tensor<3xf32>, %bias: tensor<3xf32>, %mean: tensor<3xf32>, %var: tensor<3xf32>) -> tensor<1x3x?xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%x, %scale, %bias, %mean, %var) {epsilon = 1.0e-5 : f32, momentum = 0.9 : f32} : (tensor<1x3x?xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x3x?xf32>
  return %0 : tensor<1x3x?xf32>
}

// CHECK-LABEL: func.func @batch_norm_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.BatchNormalizationInferenceMode"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @gemm_dynamic_bias(%a: tensor<2x5xf32>, %b: tensor<5x4xf32>, %c: tensor<?x4xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) : (tensor<2x5xf32>, tensor<5x4xf32>, tensor<?x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @gemm_dynamic_bias(
// CHECK: %[[RESULT:.*]] = "onnx.Gemm"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]

// -----

func.func @gemm_dynamic_output(%a: tensor<2x5xf32>, %b: tensor<5x4xf32>, %c: tensor<2x4xf32>) -> tensor<?x4xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) : (tensor<2x5xf32>, tensor<5x4xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @gemm_dynamic_output(
// CHECK: %[[RESULT:.*]] = "onnx.Gemm"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]
