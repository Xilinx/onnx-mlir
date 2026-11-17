// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,ONNX --implicit-check-not=tosa.reshape
// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,TOSA --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,TOSA --implicit-check-not='"onnx.Reshape"'

func.func @add_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @add_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.add {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @sub_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @sub_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.sub {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @pow_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @pow_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.pow {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @min_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @min_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.minimum {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @max_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @max_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.maximum {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @and_i1(%arg0: tensor<2x3x4xi1>, %arg1: tensor<4xi1>) -> tensor<2x3x4xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<2x3x4xi1>, tensor<4xi1>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @and_i1(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi1>, tensor<3xi64>) -> tensor<1x1x4xi1>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi1>) -> tensor<1x1x4xi1>
// CHECK: tosa.logical_and {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @or_i1(%arg0: tensor<2x3x4xi1>, %arg1: tensor<4xi1>) -> tensor<2x3x4xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<2x3x4xi1>, tensor<4xi1>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @or_i1(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi1>, tensor<3xi64>) -> tensor<1x1x4xi1>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi1>) -> tensor<1x1x4xi1>
// CHECK: tosa.logical_or {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @xor_i1(%arg0: tensor<2x3x4xi1>, %arg1: tensor<4xi1>) -> tensor<2x3x4xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<2x3x4xi1>, tensor<4xi1>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @xor_i1(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi1>, tensor<3xi64>) -> tensor<1x1x4xi1>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi1>) -> tensor<1x1x4xi1>
// CHECK: tosa.logical_xor {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @bitwiseand_i32(%arg0: tensor<2x3x4xi32>, %arg1: tensor<4xi32>) -> tensor<2x3x4xi32> {
  %0 = "onnx.BitwiseAnd"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<4xi32>) -> tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func.func @bitwiseand_i32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi32>, tensor<3xi64>) -> tensor<1x1x4xi32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi32>) -> tensor<1x1x4xi32>
// CHECK: tosa.bitwise_and {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi32>

// -----

func.func @bitwiseor_i32(%arg0: tensor<2x3x4xi32>, %arg1: tensor<4xi32>) -> tensor<2x3x4xi32> {
  %0 = "onnx.BitwiseOr"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<4xi32>) -> tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func.func @bitwiseor_i32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi32>, tensor<3xi64>) -> tensor<1x1x4xi32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi32>) -> tensor<1x1x4xi32>
// CHECK: tosa.bitwise_or {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi32>

// -----

func.func @bitwisexor_i32(%arg0: tensor<2x3x4xi32>, %arg1: tensor<4xi32>) -> tensor<2x3x4xi32> {
  %0 = "onnx.BitwiseXor"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<4xi32>) -> tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func.func @bitwisexor_i32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi32>, tensor<3xi64>) -> tensor<1x1x4xi32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi32>) -> tensor<1x1x4xi32>
// CHECK: tosa.bitwise_xor {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi32>

// -----

func.func @mul_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @mul_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.mul {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @div_i32(%arg0: tensor<2x3x4xi32>, %arg1: tensor<4xi32>) -> tensor<2x3x4xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<4xi32>) -> tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func.func @div_i32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xi32>, tensor<3xi64>) -> tensor<1x1x4xi32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xi32>) -> tensor<1x1x4xi32>
// CHECK: tosa.int_div {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi32>

// -----

func.func @div_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @div_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// CHECK: %[[RECIPROCAL:.*]] = tosa.reciprocal %[[ARG1]]
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[RECIPROCAL]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[RECIPROCAL]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.mul {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @equal_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xi1> {
  %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @equal_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.equal {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @greater_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xi1> {
  %0 = "onnx.Greater"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @greater_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.greater {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @greaterorequal_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xi1> {
  %0 = "onnx.GreaterOrEqual"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @greaterorequal_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.greater_equal {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @less_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @less_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.greater {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @lessorequal_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xi1> {
  %0 = "onnx.LessOrEqual"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xi1>
  return %0 : tensor<2x3x4xi1>
}

// CHECK-LABEL: func.func @lessorequal_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.greater_equal {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xi1>

// -----

func.func @prelu_f32(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @prelu_f32(
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]:
// ONNX: %[[RESHAPE:.*]] = "onnx.Reshape"(%[[ARG1]], %{{.*}}) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
// TOSA: %[[RESHAPE:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK: tosa.mul {{.*}}%[[RESHAPE]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @where(%pred: tensor<i1>, %lhs: tensor<4xf32>, %rhs: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Where"(%pred, %lhs, %rhs) : (tensor<i1>, tensor<4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @where(
// CHECK-SAME: %[[PRED_ARG:.*]]: tensor<i1>, %[[LHS_ARG:.*]]: tensor<4xf32>, %[[RHS_ARG:.*]]:
// ONNX-DAG: %[[PRED:.*]] = "onnx.Reshape"(%[[PRED_ARG]], %{{.*}}) {{.*}} -> tensor<1x1x1xi1>
// ONNX-DAG: %[[LHS:.*]] = "onnx.Reshape"(%[[LHS_ARG]], %{{.*}}) {{.*}} -> tensor<1x1x4xf32>
// TOSA-DAG: %[[PRED:.*]] = tosa.reshape %[[PRED_ARG]] {{.*}} -> tensor<1x1x1xi1>
// TOSA-DAG: %[[LHS:.*]] = tosa.reshape %[[LHS_ARG]] {{.*}} -> tensor<1x1x4xf32>
// CHECK: tosa.select %[[PRED]], %[[LHS]], %[[RHS_ARG]]
// CHECK: return {{.*}} : tensor<2x3x4xf32>

// -----

func.func @clip(%input: tensor<2x3x4xf32>, %min: tensor<f32>, %max: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Clip"(%input, %min, %max) : (tensor<2x3x4xf32>, tensor<f32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @clip(
// CHECK-SAME: %[[INPUT:.*]]: {{.*}}, %[[MIN_ARG:.*]]: tensor<f32>, %[[MAX_ARG:.*]]:
// ONNX: %[[MIN:.*]] = "onnx.Reshape"(%[[MIN_ARG]], %{{.*}}) {{.*}} -> tensor<1x1x1xf32>
// TOSA: %[[MIN:.*]] = tosa.reshape %[[MIN_ARG]] {{.*}} -> tensor<1x1x1xf32>
// CHECK: %[[MAXIMUM:.*]] = tosa.maximum %[[INPUT]], %[[MIN]]
// ONNX: %[[MAX:.*]] = "onnx.Reshape"(%[[MAX_ARG]], %{{.*}}) {{.*}} -> tensor<1x1x4xf32>
// TOSA: %[[MAX:.*]] = tosa.reshape %[[MAX_ARG]] {{.*}} -> tensor<1x1x4xf32>
// CHECK: %[[MINIMUM:.*]] = tosa.minimum %[[MAXIMUM]], %[[MAX]]
// CHECK: return %[[MINIMUM]] : tensor<2x3x4xf32>

// -----

func.func @mul_scalar_lhs(%lhs: tensor<f32>, %rhs: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Mul"(%lhs, %rhs) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_scalar_lhs(
// CHECK-SAME: %[[LHS_ARG:.*]]: tensor<f32>, %[[RHS:.*]]:
// ONNX: %[[LHS:.*]] = "onnx.Reshape"(%[[LHS_ARG]], %{{.*}}) {{.*}} -> tensor<1x1xf32>
// TOSA: %[[LHS:.*]] = tosa.reshape %[[LHS_ARG]] {{.*}} -> tensor<1x1xf32>
// CHECK: tosa.mul %[[LHS]], %[[RHS]]
// CHECK: return {{.*}} : tensor<2x3xf32>

// -----

func.func @mul_same_rank(%lhs: tensor<2x3xf32>, %rhs: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Mul"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_same_rank(
// CHECK-NOT: "onnx.Reshape"
// CHECK-NOT: tosa.reshape
// CHECK: tosa.mul
// CHECK: return {{.*}} : tensor<2x3xf32>

// -----

func.func @mul_unranked(%lhs: tensor<*xf32>, %rhs: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%lhs, %rhs) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @mul_unranked(
// CHECK: %[[RESULT:.*]] = "onnx.Mul"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]
