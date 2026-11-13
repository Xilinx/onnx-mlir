// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

func.func @gqa_compat_none_query_seqlens_separate_eps(%arg0: tensor<1x2x8xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>, %arg3: tensor<1x1x4x4xf32>, %arg4: tensor<1x1x4x4xf32>, %arg5: tensor<1xi32>, %arg6: tensor<i32>) -> (tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>) {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1:3 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0) {LayerName = "gqa", PartOfLayerName = "gqa", do_rotary = 0 : si64, domain_name = "com.microsoft", function_name = "GroupQueryAttention", k_norm_epsilon = 9.99999997E-7 : f32, kv_num_heads = 1 : si64, num_heads = 2 : si64, q_norm_epsilon = 9.99999997E-7 : f32} : (tensor<1x2x8xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none, none, none, none, none) -> (tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>)
  return %1#0, %1#1, %1#2 : tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>
}
// CHECK-LABEL: func.func @gqa_compat_none_query_seqlens_separate_eps(
// CHECK-NOT: function_name = "GroupQueryAttention"
// CHECK: "onnx.Attention"
// CHECK: return

// -----

func.func @gqa_compat_present_query_seqlens(%arg0: tensor<1x2x8xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>, %arg3: tensor<1x1x4x4xf32>, %arg4: tensor<1x1x4x4xf32>, %arg5: tensor<1xi32>, %arg6: tensor<i32>, %arg7: tensor<1xi32>) -> (tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>) {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1:3 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %0, %0, %0, %0, %0, %0, %0, %0, %0, %arg7) {do_rotary = 0 : si64, domain_name = "com.microsoft", function_name = "GroupQueryAttention", kv_num_heads = 1 : si64, num_heads = 2 : si64, q_norm_epsilon = 9.99999997E-7 : f32} : (tensor<1x2x8xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none, none, none, none, tensor<1xi32>) -> (tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>)
  return %1#0, %1#1, %1#2 : tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>
}
// CHECK-LABEL: func.func @gqa_compat_present_query_seqlens(
// CHECK-NOT: "onnx.Attention"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.Attention"
// CHECK: return

// -----

func.func @gqa_compat_present_q_norm(%arg0: tensor<1x2x8xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>, %arg3: tensor<1x1x4x4xf32>, %arg4: tensor<1x1x4x4xf32>, %arg5: tensor<1xi32>, %arg6: tensor<i32>, %arg7: tensor<4xf32>) -> (tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>) {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1:3 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %0, %0, %0, %0, %0, %0, %0, %arg7, %0, %0) {do_rotary = 0 : si64, domain_name = "com.microsoft", function_name = "GroupQueryAttention", kv_num_heads = 1 : si64, num_heads = 2 : si64, q_norm_epsilon = 9.99999997E-7 : f32} : (tensor<1x2x8xf32>, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x1x4x4xf32>, tensor<1x1x4x4xf32>, tensor<1xi32>, tensor<i32>, none, none, none, none, none, none, none, tensor<4xf32>, none, none) -> (tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>)
  return %1#0, %1#1, %1#2 : tensor<1x2x8xf32>, tensor<1x1x6x4xf32>, tensor<1x1x6x4xf32>
}
// CHECK-LABEL: func.func @gqa_compat_present_q_norm(
// CHECK-NOT: "onnx.Attention"
// CHECK: "onnx.Custom"
// CHECK-NOT: "onnx.Attention"
// CHECK: return

// -----
