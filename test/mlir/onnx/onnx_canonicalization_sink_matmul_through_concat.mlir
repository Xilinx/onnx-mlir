// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s | FileCheck %s

func.func @batch_axis(%a0: tensor<1x8x3xf32>, %a1: tensor<1x8x3xf32>,
    %b0: tensor<1x3x6xf32>, %b1: tensor<1x3x6xf32>) -> tensor<2x8x6xf32> {
  %p = "onnx.MatMul"(%a0, %b0) : (tensor<1x8x3xf32>, tensor<1x3x6xf32>) -> tensor<1x8x6xf32>
  %q = "onnx.MatMul"(%a1, %b1) : (tensor<1x8x3xf32>, tensor<1x3x6xf32>) -> tensor<1x8x6xf32>
  %result = "onnx.Concat"(%p, %q) {axis = 0 : si64} : (tensor<1x8x6xf32>, tensor<1x8x6xf32>) -> tensor<2x8x6xf32>
  return %result : tensor<2x8x6xf32>
}
// CHECK-LABEL: func.func @batch_axis(
// CHECK-SAME: [[A0:%[^:]+]]: tensor<1x8x3xf32>, [[A1:%[^:]+]]: tensor<1x8x3xf32>, [[B0:%[^:]+]]: tensor<1x3x6xf32>, [[B1:%[^:]+]]: tensor<1x3x6xf32>
// CHECK-NOT: "onnx.MatMul"
// CHECK-DAG: [[LHS:%[^ ]+]] = "onnx.Concat"([[A0]], [[A1]]) {axis = 0 : si64} : (tensor<1x8x3xf32>, tensor<1x8x3xf32>) -> tensor<2x8x3xf32>
// CHECK-DAG: [[RHS:%[^ ]+]] = "onnx.Concat"([[B0]], [[B1]]) {axis = 0 : si64} : (tensor<1x3x6xf32>, tensor<1x3x6xf32>) -> tensor<2x3x6xf32>
// CHECK-NOT: "onnx.MatMul"
// CHECK: [[RESULT:%[^ ]+]] = "onnx.MatMul"([[LHS]], [[RHS]]) : (tensor<2x8x3xf32>, tensor<2x3x6xf32>) -> tensor<2x8x6xf32>
// CHECK-NEXT: return [[RESULT]] : tensor<2x8x6xf32>

func.func @negative_batch_axis(%a0: tensor<2x8x3xf32>, %a1: tensor<2x8x3xf32>,
    %b0: tensor<2x3x6xf32>, %b1: tensor<2x3x6xf32>) -> tensor<4x8x6xf32> {
  %p = "onnx.MatMul"(%a0, %b0) : (tensor<2x8x3xf32>, tensor<2x3x6xf32>) -> tensor<2x8x6xf32>
  %q = "onnx.MatMul"(%a1, %b1) : (tensor<2x8x3xf32>, tensor<2x3x6xf32>) -> tensor<2x8x6xf32>
  %result = "onnx.Concat"(%p, %q) {axis = -3 : si64} : (tensor<2x8x6xf32>, tensor<2x8x6xf32>) -> tensor<4x8x6xf32>
  return %result : tensor<4x8x6xf32>
}
// CHECK-LABEL: func.func @negative_batch_axis(
// CHECK-SAME: [[A0:%[^:]+]]: tensor<2x8x3xf32>, [[A1:%[^:]+]]: tensor<2x8x3xf32>, [[B0:%[^:]+]]: tensor<2x3x6xf32>, [[B1:%[^:]+]]: tensor<2x3x6xf32>
// CHECK-NOT: "onnx.MatMul"
// CHECK-DAG: [[LHS:%[^ ]+]] = "onnx.Concat"([[A0]], [[A1]]) {axis = 0 : si64} : (tensor<2x8x3xf32>, tensor<2x8x3xf32>) -> tensor<4x8x3xf32>
// CHECK-DAG: [[RHS:%[^ ]+]] = "onnx.Concat"([[B0]], [[B1]]) {axis = 0 : si64} : (tensor<2x3x6xf32>, tensor<2x3x6xf32>) -> tensor<4x3x6xf32>
// CHECK-NOT: "onnx.MatMul"
// CHECK: [[RESULT:%[^ ]+]] = "onnx.MatMul"([[LHS]], [[RHS]]) : (tensor<4x8x3xf32>, tensor<4x3x6xf32>) -> tensor<4x8x6xf32>
// CHECK-NEXT: return [[RESULT]] : tensor<4x8x6xf32>

func.func @row_axis(%a0: tensor<1x8x3xf32>, %a1: tensor<1x8x3xf32>,
    %b0: tensor<1x3x6xf32>, %b1: tensor<1x3x6xf32>) -> tensor<1x16x6xf32> {
  %p = "onnx.MatMul"(%a0, %b0) : (tensor<1x8x3xf32>, tensor<1x3x6xf32>) -> tensor<1x8x6xf32>
  %q = "onnx.MatMul"(%a1, %b1) : (tensor<1x8x3xf32>, tensor<1x3x6xf32>) -> tensor<1x8x6xf32>
  %result = "onnx.Concat"(%p, %q) {axis = 1 : si64} : (tensor<1x8x6xf32>, tensor<1x8x6xf32>) -> tensor<1x16x6xf32>
  return %result : tensor<1x16x6xf32>
}
// CHECK-LABEL: func.func @row_axis(
// CHECK-SAME: [[A0:%[^:]+]]: tensor<1x8x3xf32>, [[A1:%[^:]+]]: tensor<1x8x3xf32>, [[B0:%[^:]+]]: tensor<1x3x6xf32>, [[B1:%[^:]+]]: tensor<1x3x6xf32>
// CHECK-DAG: [[P:%[^ ]+]] = "onnx.MatMul"([[A0]], [[B0]])
// CHECK-DAG: [[Q:%[^ ]+]] = "onnx.MatMul"([[A1]], [[B1]])
// CHECK: [[RESULT:%[^ ]+]] = "onnx.Concat"([[P]], [[Q]]) {axis = 1 : si64}
// CHECK: return [[RESULT]] : tensor<1x16x6xf32>

func.func @broadcast_batch(%a0: tensor<1x8x3xf32>, %a1: tensor<1x8x3xf32>,
    %b0: tensor<2x3x6xf32>, %b1: tensor<2x3x6xf32>) -> tensor<4x8x6xf32> {
  %p = "onnx.MatMul"(%a0, %b0) : (tensor<1x8x3xf32>, tensor<2x3x6xf32>) -> tensor<2x8x6xf32>
  %q = "onnx.MatMul"(%a1, %b1) : (tensor<1x8x3xf32>, tensor<2x3x6xf32>) -> tensor<2x8x6xf32>
  %result = "onnx.Concat"(%p, %q) {axis = 0 : si64} : (tensor<2x8x6xf32>, tensor<2x8x6xf32>) -> tensor<4x8x6xf32>
  return %result : tensor<4x8x6xf32>
}
// CHECK-LABEL: func.func @broadcast_batch(
// CHECK-SAME: [[A0:%[^:]+]]: tensor<1x8x3xf32>, [[A1:%[^:]+]]: tensor<1x8x3xf32>, [[B0:%[^:]+]]: tensor<2x3x6xf32>, [[B1:%[^:]+]]: tensor<2x3x6xf32>
// CHECK-DAG: [[P:%[^ ]+]] = "onnx.MatMul"([[A0]], [[B0]])
// CHECK-DAG: [[Q:%[^ ]+]] = "onnx.MatMul"([[A1]], [[B1]])
// CHECK: [[RESULT:%[^ ]+]] = "onnx.Concat"([[P]], [[Q]]) {axis = 0 : si64}
// CHECK: return [[RESULT]] : tensor<4x8x6xf32>

func.func @shared_product(%a0: tensor<1x8x3xf32>, %a1: tensor<1x8x3xf32>,
    %b0: tensor<1x3x6xf32>, %b1: tensor<1x3x6xf32>) -> (tensor<2x8x6xf32>, tensor<1x8x6xf32>) {
  %p = "onnx.MatMul"(%a0, %b0) : (tensor<1x8x3xf32>, tensor<1x3x6xf32>) -> tensor<1x8x6xf32>
  %q = "onnx.MatMul"(%a1, %b1) : (tensor<1x8x3xf32>, tensor<1x3x6xf32>) -> tensor<1x8x6xf32>
  %result = "onnx.Concat"(%p, %q) {axis = 0 : si64} : (tensor<1x8x6xf32>, tensor<1x8x6xf32>) -> tensor<2x8x6xf32>
  return %result, %p : tensor<2x8x6xf32>, tensor<1x8x6xf32>
}
// CHECK-LABEL: func.func @shared_product(
// CHECK-SAME: [[A0:%[^:]+]]: tensor<1x8x3xf32>, [[A1:%[^:]+]]: tensor<1x8x3xf32>, [[B0:%[^:]+]]: tensor<1x3x6xf32>, [[B1:%[^:]+]]: tensor<1x3x6xf32>
// CHECK-DAG: [[P:%[^ ]+]] = "onnx.MatMul"([[A0]], [[B0]])
// CHECK-DAG: [[Q:%[^ ]+]] = "onnx.MatMul"([[A1]], [[B1]])
// CHECK: [[RESULT:%[^ ]+]] = "onnx.Concat"([[P]], [[Q]]) {axis = 0 : si64}
// CHECK: return [[RESULT]], [[P]] : tensor<2x8x6xf32>
