// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt %s --constprop-onnx="enable-qdq form-deferred-foldable-expressions" | FileCheck %s
// RUN: onnx-mlir-opt %s --onnx-form-deferred-foldable-expressions="enable-qdq" --constprop-onnx | FileCheck %s
// RUN: onnx-mlir-opt %s --onnx-form-deferred-foldable-expressions --constprop-onnx="enable-qdq" | FileCheck %s
// RUN: onnx-mlir-opt %s --onnx-form-deferred-foldable-expressions | FileCheck %s --check-prefix=DISABLED

// Start with a retained DQ and extend it with a Transpose. An outer algebraic
// rewrite must see the yielded DQ when deciding whether to commute Add.
func.func @transpose_deferred_dq(%runtime: tensor<2x1xf32>) -> tensor<2x1xf32> {
  %w = "onnx.InitializerSource"() {kind = "onnx_model", name = "w", path = "unused.onnx", scope = []} : () -> tensor<1x2xi8>
  %dq = "onnx.DeferredFoldable"(%w) ({
  ^bb0(%data: tensor<1x2xi8>):
    %scale = onnx.Constant dense<0.5> : tensor<f32>
    %zp = onnx.Constant dense<0> : tensor<i8>
    %d = "onnx.DequantizeLinear"(%data, %scale, %zp) : (tensor<1x2xi8>, tensor<f32>, tensor<i8>) -> tensor<1x2xf32>
    onnx.Yield %d : tensor<1x2xf32>
  }) : (tensor<1x2xi8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %a = "onnx.Add"(%t, %runtime) : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  return %a : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @transpose_deferred_dq(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[W:.*]] = "onnx.DeferredFoldable"
// CHECK: %[[T:.*]] = "onnx.Transpose"{{.*}} -> tensor<2x1xi8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[T]],
// CHECK: onnx.Yield %[[DQ]]
// CHECK: "onnx.Add"(%[[W]], %[[RUNTIME]])

// DISABLED-LABEL: func.func @transpose_deferred_dq(
// DISABLED: %[[DQ:.*]] = "onnx.DequantizeLinear"
// DISABLED: %[[T:.*]] = "onnx.Transpose"(%[[DQ]])
// DISABLED: onnx.Yield %[[T]]

// A per-axis DQ would need its axis adjusted after the transpose.
func.func @transpose_per_axis_dq() -> tensor<2x1xf32> {
  %w = "onnx.InitializerSource"() {kind = "onnx_model", name = "w", path = "unused.onnx", scope = []} : () -> tensor<1x2xi8>
  %scale = onnx.Constant dense<[0.5, 0.25]> : tensor<2xf32>
  %zp = onnx.Constant dense<0> : tensor<2xi8>
  %dq = "onnx.DequantizeLinear"(%w, %scale, %zp) {axis = 1 : si64} : (tensor<1x2xi8>, tensor<2xf32>, tensor<2xi8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  return %t : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @transpose_per_axis_dq(
// CHECK: %[[D:.*]] = "onnx.DequantizeLinear"
// CHECK: %[[T:.*]] = "onnx.Transpose"(%[[D]])
// CHECK: onnx.Yield %[[T]]
