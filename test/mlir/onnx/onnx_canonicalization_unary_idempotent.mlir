// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

// -----

func.func @abs_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Abs"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Abs"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @abs_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK: [[INNER:%[^ ]+]] = "onnx.Abs"([[X]])
// CHECK-NOT: "onnx.Abs"
// CHECK: return [[INNER]]

// -----

func.func @abs_i32(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Abs"(%x) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %1 = "onnx.Abs"(%0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @abs_i32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xi32>
// CHECK: [[INNER:%[^ ]+]] = "onnx.Abs"([[X]])
// CHECK-NOT: "onnx.Abs"
// CHECK: return [[INNER]]

// -----

func.func @relu_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Relu"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Relu"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @relu_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK: [[INNER:%[^ ]+]] = "onnx.Relu"([[X]])
// CHECK-NOT: "onnx.Relu"
// CHECK: return [[INNER]]

// -----

func.func @floor_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Floor"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Floor"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @floor_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK: [[INNER:%[^ ]+]] = "onnx.Floor"([[X]])
// CHECK-NOT: "onnx.Floor"
// CHECK: return [[INNER]]

// -----

func.func @ceil_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Ceil"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Ceil"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @ceil_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK: [[INNER:%[^ ]+]] = "onnx.Ceil"([[X]])
// CHECK-NOT: "onnx.Ceil"
// CHECK: return [[INNER]]

// -----

func.func @round_f32(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Round"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Round"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @round_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK: [[INNER:%[^ ]+]] = "onnx.Round"([[X]])
// CHECK-NOT: "onnx.Round"
// CHECK: return [[INNER]]

// -----

// The input type may differ: replacing with the inner result preserves refinement.
func.func @abs_refined_inner(%x: tensor<?x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  %0 = "onnx.Abs"(%x) : (tensor<?x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Abs"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: func.func @abs_refined_inner(
// CHECK: [[INNER:%[^ ]+]] = "onnx.Abs"{{.*}} : (tensor<?x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Abs"
// CHECK: return [[INNER]], [[INNER]]

// -----

// Preserve shape refinement introduced by the outer operation.
func.func @abs_refined_outer(%x: tensor<?x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Abs"(%x) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %1 = "onnx.Abs"(%0) : (tensor<?x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @abs_refined_outer(
// CHECK: [[INNER:%[^ ]+]] = "onnx.Abs"
// CHECK: [[OUTER:%[^ ]+]] = "onnx.Abs"([[INNER]])
// CHECK: return [[OUTER]]

// -----

func.func @not_cancel(%x: tensor<2x3xi1>) -> tensor<2x3xi1> {
  %0 = "onnx.Not"(%x) : (tensor<2x3xi1>) -> tensor<2x3xi1>
  %1 = "onnx.Not"(%0) : (tensor<2x3xi1>) -> tensor<2x3xi1>
  return %1 : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @not_cancel(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xi1>
// CHECK-NOT: "onnx.Not"
// CHECK: return [[X]]

// -----

func.func @not_refinement(%x: tensor<?x3xi1>) -> tensor<2x3xi1> {
  %0 = "onnx.Not"(%x) : (tensor<?x3xi1>) -> tensor<2x3xi1>
  %1 = "onnx.Not"(%0) : (tensor<2x3xi1>) -> tensor<2x3xi1>
  return %1 : tensor<2x3xi1>
}

// CHECK-LABEL: func.func @not_refinement(
// CHECK: [[INNER:%[^ ]+]] = "onnx.Not"
// CHECK: [[OUTER:%[^ ]+]] = "onnx.Not"([[INNER]])
// CHECK: return [[OUTER]]

// -----

func.func @bitwisenot_cancel(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.BitwiseNot"(%x) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %1 = "onnx.BitwiseNot"(%0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @bitwisenot_cancel(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xi32>
// CHECK-NOT: "onnx.BitwiseNot"
// CHECK: return [[X]]

// -----

func.func @bitwisenot_refinement(%x: tensor<?x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.BitwiseNot"(%x) : (tensor<?x3xi32>) -> tensor<2x3xi32>
  %1 = "onnx.BitwiseNot"(%0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @bitwisenot_refinement(
// CHECK: [[INNER:%[^ ]+]] = "onnx.BitwiseNot"
// CHECK: [[OUTER:%[^ ]+]] = "onnx.BitwiseNot"([[INNER]])
// CHECK: return [[OUTER]]

// -----

// Preserve the existing exclusion for quantized Abs.
func.func @abs_quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.02>>) -> tensor<2x!quant.uniform<i8:f32, 0.02>> {
  %0 = "onnx.Abs"(%x) : (tensor<2x!quant.uniform<i8:f32, 0.02>>) -> tensor<2x!quant.uniform<i8:f32, 0.02>>
  %1 = "onnx.Abs"(%0) : (tensor<2x!quant.uniform<i8:f32, 0.02>>) -> tensor<2x!quant.uniform<i8:f32, 0.02>>
  return %1 : tensor<2x!quant.uniform<i8:f32, 0.02>>
}

// CHECK-LABEL: func.func @abs_quantized(
// CHECK: [[INNER:%[^ ]+]] = "onnx.Abs"
// CHECK: [[OUTER:%[^ ]+]] = "onnx.Abs"([[INNER]])
// CHECK: return [[OUTER]]

// -----

func.func @floor_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Floor"(%x) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "onnx.Floor"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

// CHECK-LABEL: func.func @floor_unranked(
// CHECK: [[INNER:%[^ ]+]] = "onnx.Floor"
// CHECK-NOT: "onnx.Floor"
// CHECK: return [[INNER]]
