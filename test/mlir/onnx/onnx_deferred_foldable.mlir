// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --onnx-form-deferred-foldable-expressions %s | FileCheck %s
// RUN: onnx-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=CANON

func.func @retain_expression(%runtime: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %bias = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %deferred = "onnx.Add"(%transposed, %bias) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %mixed = "onnx.Sub"(%runtime, %deferred) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %deferred, %mixed : tensor<2x2xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @retain_expression
// CHECK-SAME:  (%[[RUNTIME:.+]]: tensor<2x2xf32>)
// CHECK-DAG:     %[[WEIGHT:.+]] = "onnx.InitializerSource"()
// CHECK-DAG:     %[[BIAS:.+]] = onnx.Constant dense<1.000000e+00>
// CHECK-NOT:     "onnx.DeferredFoldable"
// CHECK:         %[[DEFERRED:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[BIAS]]) ({
// CHECK:         ^bb0(%[[CAPTURED_WEIGHT:.+]]: tensor<2x2xf32>, %[[CAPTURED_BIAS:.+]]: tensor<2x2xf32>):
// CHECK:           %[[TRANSPOSE:.+]] = "onnx.Transpose"(%[[CAPTURED_WEIGHT]])
// CHECK:           %[[ADD:.+]] = "onnx.Add"(%[[TRANSPOSE]], %[[CAPTURED_BIAS]])
// CHECK:           onnx.DeferredYield %[[ADD]] : tensor<2x2xf32>
// CHECK:         }) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:         %[[MIXED:.+]] = "onnx.Sub"(%[[RUNTIME]], %[[DEFERRED]])
// CHECK:         onnx.Return %[[DEFERRED]], %[[MIXED]]

// A non-deterministic operation does not have the Pure trait and must remain
// visible even when its input is deferred.
func.func @do_not_retain_nondeterministic_expression() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "random_input", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %random = "onnx.RandomNormalLike"(%weight) {dtype = 1 : si64, mean = 0.0 : f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %random : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @do_not_retain_nondeterministic_expression
// CHECK:         %[[SOURCE:.+]] = "onnx.InitializerSource"()
// CHECK-NEXT:    %[[RANDOM:.+]] = "onnx.RandomNormalLike"(%[[SOURCE]])
// CHECK-NEXT:    onnx.Return %[[RANDOM]]

// Generic cleanup is not the recipe evaluation mechanism, but a
// semantics-preserving canonicalization may eliminate an identity terminal.
// The recipe remains valid and directly yields its capture in that case.
func.func @canonicalized_identity_recipe() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%captured: tensor<2x2xf32>):
    %identity = "onnx.Identity"(%captured) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %identity : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>
}

// CANON-LABEL: func.func @canonicalized_identity_recipe
// CANON:         %[[SOURCE:.+]] = "onnx.InitializerSource"()
// CANON:         %[[DEFERRED:.+]] = "onnx.DeferredFoldable"(%[[SOURCE]]) ({
// CANON-NOT:       "onnx.Identity"
// CANON:         ^bb0(%[[CAPTURE:.+]]: tensor<2x2xf32>):
// CANON-NEXT:      onnx.DeferredYield %[[CAPTURE]]
// CANON:         onnx.Return %[[DEFERRED]]
