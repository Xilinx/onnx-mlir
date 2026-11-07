// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --onnx-form-deferred-foldable-expressions %s | FileCheck %s
// RUN: onnx-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=CANON
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=false" %s | FileCheck %s --check-prefix=NOFORM

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

// NOFORM-LABEL: func.func @retain_expression
// NOFORM-NOT:     "onnx.DeferredFoldable"
// NOFORM:         %[[TRANSPOSE:.+]] = "onnx.Transpose"
// NOFORM-NEXT:    %[[ADD:.+]] = "onnx.Add"
// NOFORM-NEXT:    %[[SUB:.+]] = "onnx.Sub"
// NOFORM-NEXT:    onnx.Return %[[ADD]], %[[SUB]]

// An ordinary constant expression has no unavailable source and must remain
// available to normal constant propagation rather than becoming deferred.
func.func @do_not_retain_dense_constant_expression() -> tensor<2x2xf32> {
  %constant = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %transposed = "onnx.Transpose"(%constant) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %transposed : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @do_not_retain_dense_constant_expression
// CHECK-NOT:     "onnx.DeferredFoldable"
// CHECK:         %[[CONSTANT:.+]] = onnx.Constant
// CHECK-NEXT:    %[[TRANSPOSE:.+]] = "onnx.Transpose"(%[[CONSTANT]])
// CHECK-NEXT:    onnx.Return %[[TRANSPOSE]]

// A pure multi-result operation is retained as one single-result recipe per
// used result so each producer remains directly classifiable.
func.func @retain_multi_result_expression() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 48 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x6xf32>
  %splits = onnx.Constant dense<2> : tensor<3xi64>
  %parts:3 = "onnx.Split"(%weight, %splits) {axis = 1 : si64} : (tensor<2x6xf32>, tensor<3xi64>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  onnx.Return %parts#0, %parts#1, %parts#2 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @retain_multi_result_expression
// CHECK-DAG:     %[[WEIGHT:.+]] = "onnx.InitializerSource"()
// CHECK-DAG:     %[[SPLITS:.+]] = onnx.Constant dense<2>
// CHECK:         %[[PART0:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[SPLITS]]) ({
// CHECK:           %[[SPLIT0:.+]]:3 = "onnx.Split"
// CHECK:           onnx.DeferredYield %[[SPLIT0]]#0
// CHECK:         %[[PART1:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[SPLITS]]) ({
// CHECK:           %[[SPLIT1:.+]]:3 = "onnx.Split"
// CHECK:           onnx.DeferredYield %[[SPLIT1]]#1
// CHECK:         %[[PART2:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[SPLITS]]) ({
// CHECK:           %[[SPLIT2:.+]]:3 = "onnx.Split"
// CHECK:           onnx.DeferredYield %[[SPLIT2]]#2
// CHECK:         onnx.Return %[[PART0]], %[[PART1]], %[[PART2]]

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
// Once the recipe directly yields a capture, it is redundant and is removed.
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
// CANON-NOT:     "onnx.DeferredFoldable"
// CANON-NOT:     "onnx.Identity"
// CANON:         onnx.Return %[[SOURCE]]

// A retained recipe remains valid when later dialect conversion lowers an
// operation in its body.
func.func @lowered_recipe_operation() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %reshaped = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 2>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %reshaped : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
