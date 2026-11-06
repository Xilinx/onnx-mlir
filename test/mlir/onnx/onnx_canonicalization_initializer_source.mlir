// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s | FileCheck %s

// Propagating the attention scale into the projection weights is an algebraic
// rewrite. It must build a deferred foldable expression when the weights are
// initializer sources, without reading their payloads.
func.func @propagate_attention_scale_into_deferred_query(
    %query: tensor<1x2x4xf32>, %key: tensor<1x2x4xf32>)
    -> tensor<1x2x2x2xf32> {
  %query_weight = "onnx.InitializerSource"() {kind = "onnx_model", name = "query_weight", path = "/model.onnx", scope = []} : () -> tensor<4x4xf32>
  %query_bias = "onnx.InitializerSource"() {kind = "onnx_model", name = "query_bias", path = "/model.onnx", scope = []} : () -> tensor<4xf32>
  %key_weight = "onnx.InitializerSource"() {kind = "onnx_model", name = "key_weight", path = "/model.onnx", scope = []} : () -> tensor<4x4xf32>
  %key_bias = "onnx.InitializerSource"() {kind = "onnx_model", name = "key_bias", path = "/model.onnx", scope = []} : () -> tensor<4xf32>
  %shape = onnx.Constant dense<[1, 2, 2, 2]> : tensor<4xi64>
  %scale = onnx.Constant dense<2.000000e+00> : tensor<f32>

  %query_matmul = "onnx.MatMul"(%query, %query_weight) : (tensor<1x2x4xf32>, tensor<4x4xf32>) -> tensor<1x2x4xf32>
  %query_add = "onnx.Add"(%query_matmul, %query_bias) : (tensor<1x2x4xf32>, tensor<4xf32>) -> tensor<1x2x4xf32>
  %query_reshape = "onnx.Reshape"(%query_add, %shape) {allowzero = 0 : si64} : (tensor<1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x2xf32>
  %query_transpose = "onnx.Transpose"(%query_reshape) {perm = [0, 2, 1, 3]} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>

  %key_matmul = "onnx.MatMul"(%key, %key_weight) : (tensor<1x2x4xf32>, tensor<4x4xf32>) -> tensor<1x2x4xf32>
  %key_add = "onnx.Add"(%key_matmul, %key_bias) : (tensor<1x2x4xf32>, tensor<4xf32>) -> tensor<1x2x4xf32>
  %key_reshape = "onnx.Reshape"(%key_add, %shape) {allowzero = 0 : si64} : (tensor<1x2x4xf32>, tensor<4xi64>) -> tensor<1x2x2x2xf32>
  %key_transpose = "onnx.Transpose"(%key_reshape) {perm = [0, 2, 3, 1]} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>

  %scores = "onnx.MatMul"(%query_transpose, %key_transpose) : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %scaled_scores = "onnx.Div"(%scores, %scale) : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2x2x2xf32>
  onnx.Return %scaled_scores : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @propagate_attention_scale_into_deferred_query
// CHECK-SAME: ([[QUERY:%.+]]: tensor<1x2x4xf32>, [[KEY:%.+]]: tensor<1x2x4xf32>)
// CHECK-DAG: [[QUERY_WEIGHT:%.+]] = "onnx.InitializerSource"() {{.*}}name = "query_weight"
// CHECK-DAG: [[QUERY_BIAS:%.+]] = "onnx.InitializerSource"() {{.*}}name = "query_bias"
// CHECK-DAG: [[KEY_WEIGHT:%.+]] = "onnx.InitializerSource"() {{.*}}name = "key_weight"
// CHECK-DAG: [[KEY_BIAS:%.+]] = "onnx.InitializerSource"() {{.*}}name = "key_bias"
// CHECK-DAG: [[SCALE:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK: [[SCALED_QUERY_WEIGHT:%.+]] = "onnx.Div"([[QUERY_WEIGHT]], [[SCALE]])
// CHECK-NEXT: [[QUERY_MATMUL:%.+]] = "onnx.MatMul"([[QUERY]], [[SCALED_QUERY_WEIGHT]])
// CHECK-NEXT: [[SCALED_QUERY_BIAS:%.+]] = "onnx.Div"([[QUERY_BIAS]], [[SCALE]])
// CHECK-NEXT: "onnx.Add"([[QUERY_MATMUL]], [[SCALED_QUERY_BIAS]])
// CHECK: "onnx.MatMul"([[KEY]], [[KEY_WEIGHT]])
// CHECK: "onnx.Add"({{.*}}, [[KEY_BIAS]])
// CHECK-NOT: "onnx.Div"({{.*}}tensor<1x2x2x2xf32>
