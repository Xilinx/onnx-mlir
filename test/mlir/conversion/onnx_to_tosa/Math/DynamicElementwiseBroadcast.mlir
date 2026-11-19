// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not=shape. --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,EXCLUDED --implicit-check-not=shape. --implicit-check-not=tosa.reshape
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not=shape. --implicit-check-not='"onnx.Reshape"'

// Elementwise ops lower with dynamic or unranked shapes when rank broadcasting
// reshapes only static operands.
func.func @add_dynamic_equal_rank(%lhs: tensor<?x3xf32>, %rhs: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Add"(%lhs, %rhs) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @add_dynamic_equal_rank(
// CHECK-SAME: %[[LHS:.*]]: tensor<?x3xf32>, %[[RHS:.*]]: tensor<?x3xf32>
// CHECK: %[[RESULT:.*]] = tosa.add %[[LHS]], %[[RHS]] : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @add_dynamic_output(%lhs: tensor<2x3xf32>, %rhs: tensor<2x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Add"(%lhs, %rhs) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @add_dynamic_output(
// CHECK-SAME: %[[LHS:.*]]: tensor<2x3xf32>, %[[RHS:.*]]: tensor<2x3xf32>
// CHECK: %[[RESULT:.*]] = tosa.add %[[LHS]], %[[RHS]] : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<?x3xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @add_unranked(%lhs: tensor<*xf32>, %rhs: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%lhs, %rhs) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @add_unranked(
// CHECK-SAME: %[[LHS:.*]]: tensor<*xf32>, %[[RHS:.*]]: tensor<3xf32>
// CHECK: %[[RESULT:.*]] = tosa.add %[[LHS]], %[[RHS]] : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @add_dynamic_static_broadcast(%lhs: tensor<1x?x3xf32>, %rhs: tensor<3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Add"(%lhs, %rhs) : (tensor<1x?x3xf32>, tensor<3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @add_dynamic_static_broadcast(
// CHECK-SAME: %[[LHS:.*]]: tensor<1x?x3xf32>, %[[RHS:.*]]: tensor<3xf32>
// LOWERED: %[[RESHAPED:.*]] = tosa.reshape %[[RHS]] {new_shape = array<i64: 1, 1, 3>}
// EXCLUDED: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[1, 1, 3]> : tensor<3xi64>
// EXCLUDED: %[[RESHAPED:.*]] = "onnx.Reshape"(%[[RHS]], %[[SHAPE]]) {{.*}} -> tensor<1x1x3xf32>
// CHECK: %[[RESULT:.*]] = tosa.add %[[LHS]], %[[RESHAPED]] : (tensor<1x?x3xf32>, tensor<1x1x3xf32>) -> tensor<1x?x3xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @where_dynamic_static_broadcast(%condition: tensor<1x?x3xi1>, %x: tensor<3xf32>, %y: tensor<1x?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Where"(%condition, %x, %y) : (tensor<1x?x3xi1>, tensor<3xf32>, tensor<1x?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @where_dynamic_static_broadcast(
// CHECK-SAME: %[[CONDITION:.*]]: tensor<1x?x3xi1>, %[[X:.*]]: tensor<3xf32>, %[[Y:.*]]: tensor<1x?x3xf32>
// LOWERED: %[[RESHAPED:.*]] = tosa.reshape %[[X]] {new_shape = array<i64: 1, 1, 3>}
// EXCLUDED: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<[1, 1, 3]> : tensor<3xi64>
// EXCLUDED: %[[RESHAPED:.*]] = "onnx.Reshape"(%[[X]], %[[SHAPE]]) {{.*}} -> tensor<1x1x3xf32>
// CHECK: %[[RESULT:.*]] = tosa.select %[[CONDITION]], %[[RESHAPED]], %[[Y]] : (tensor<1x?x3xi1>, tensor<1x1x3xf32>, tensor<1x?x3xf32>) -> tensor<1x?x3xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @leaky_relu_dynamic(%x: tensor<1x?x3xf32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.LeakyRelu"(%x) : (tensor<1x?x3xf32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @leaky_relu_dynamic(
// CHECK-SAME: %[[X:.*]]: tensor<1x?x3xf32>
// CHECK: %[[MUL:.*]] = tosa.mul %[[X]], {{.*}} -> tensor<1x?x3xf32>
// CHECK: %[[GE:.*]] = tosa.greater_equal %[[X]], {{.*}} -> tensor<1x?x3xi1>
// CHECK: %[[RESULT:.*]] = tosa.select %[[GE]], %[[X]], %[[MUL]] {{.*}} -> tensor<1x?x3xf32>
// CHECK: return %[[RESULT]]

// -----

func.func @clip_dynamic(%x: tensor<1x?x3xf32>, %min: tensor<f32>, %max: tensor<f32>) -> tensor<1x?x3xf32> {
  %0 = "onnx.Clip"(%x, %min, %max) : (tensor<1x?x3xf32>, tensor<f32>, tensor<f32>) -> tensor<1x?x3xf32>
  return %0 : tensor<1x?x3xf32>
}

// CHECK-LABEL: func.func @clip_dynamic(
// CHECK-SAME: %[[X:.*]]: tensor<1x?x3xf32>, %[[MIN:.*]]: tensor<f32>, %[[MAX:.*]]: tensor<f32>
// LOWERED: %[[MIN_RESHAPED:.*]] = tosa.reshape %[[MIN]] {new_shape = array<i64: 1, 1, 1>}
// EXCLUDED: %[[SHAPE:.*]] = "tosa.const"() {{.*}}dense<1> : tensor<3xi64>
// EXCLUDED: %[[MIN_RESHAPED:.*]] = "onnx.Reshape"(%[[MIN]], %[[SHAPE]]) {{.*}} -> tensor<1x1x1xf32>
// CHECK: %[[MAXIMUM:.*]] = tosa.maximum %[[X]], %[[MIN_RESHAPED]] : (tensor<1x?x3xf32>, tensor<1x1x1xf32>) -> tensor<1x?x3xf32>
// LOWERED: %[[MAX_RESHAPED:.*]] = tosa.reshape %[[MAX]] {new_shape = array<i64: 1, 1, 1>}
// EXCLUDED: %[[MAX_RESHAPED:.*]] = "onnx.Reshape"(%[[MAX]], %[[SHAPE]]) {{.*}} -> tensor<1x1x1xf32>
// CHECK: %[[RESULT:.*]] = tosa.minimum %[[MAXIMUM]], %[[MAX_RESHAPED]] : (tensor<1x?x3xf32>, tensor<1x1x1xf32>) -> tensor<1x?x3xf32>
// CHECK: return %[[RESULT]]
