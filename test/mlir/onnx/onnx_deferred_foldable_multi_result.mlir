// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt %s | FileCheck %s
// RUN: onnx-mlir-opt %s --canonicalize | FileCheck %s --check-prefix=CANON

func.func @different_result_types() -> (tensor<4x2xi8>, tensor<f32>, tensor<i8>) {
  %w = "onnx.InitializerSource"() {kind = "external_data", length = 8 : i64, name = "w", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x4xi8>
  %values:3 = "onnx.DeferredFoldable"(%w) ({
  ^bb0(%a: tensor<2x4xi8>):
    %t = "onnx.Transpose"(%a) {perm = [1, 0]} : (tensor<2x4xi8>) -> tensor<4x2xi8>
    %s = onnx.Constant dense<0.125> : tensor<f32>
    %z = onnx.Constant dense<0> : tensor<i8>
    onnx.Yield %t, %s, %z : tensor<4x2xi8>, tensor<f32>, tensor<i8>
  }) : (tensor<2x4xi8>) -> (tensor<4x2xi8>, tensor<f32>, tensor<i8>)
  return %values#0, %values#1, %values#2 : tensor<4x2xi8>, tensor<f32>, tensor<i8>
}

// CHECK-LABEL: func.func @different_result_types
// CHECK: %[[VALUES:.+]]:3 = "onnx.DeferredFoldable"
// CHECK: onnx.Yield {{.*}} : tensor<4x2xi8>, tensor<f32>, tensor<i8>
// CHECK: return %[[VALUES]]#0, %[[VALUES]]#1, %[[VALUES]]#2
// CANON-LABEL: func.func @different_result_types
// CANON: "onnx.DeferredFoldable"

func.func @forward_capture_twice() -> (tensor<2x4xi8>, tensor<2x4xi8>) {
  %w = "onnx.InitializerSource"() {kind = "external_data", length = 8 : i64, name = "w", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x4xi8>
  %values:2 = "onnx.DeferredFoldable"(%w) ({
  ^bb0(%a: tensor<2x4xi8>):
    onnx.Yield %a, %a : tensor<2x4xi8>, tensor<2x4xi8>
  }) : (tensor<2x4xi8>) -> (tensor<2x4xi8>, tensor<2x4xi8>)
  return %values#0, %values#1 : tensor<2x4xi8>, tensor<2x4xi8>
}

// CHECK-LABEL: func.func @forward_capture_twice
// CHECK: %[[VALUES:.+]]:2 = "onnx.DeferredFoldable"
// CHECK: return %[[VALUES]]#0, %[[VALUES]]#1
// CANON-LABEL: func.func @forward_capture_twice
// CANON: %[[W:.+]] = "onnx.InitializerSource"
// CANON-NOT: "onnx.DeferredFoldable"
// CANON: return %[[W]], %[[W]]
