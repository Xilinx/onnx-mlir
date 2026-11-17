// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --onnx-form-deferred-foldable-expressions %s | FileCheck %s
// RUN: onnx-mlir-opt --onnx-form-deferred-foldable-expressions --onnx-infer-tensornames %s | FileCheck %s --check-prefix=NAMES
// RUN: onnx-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=CANON
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=false" %s | FileCheck %s --check-prefix=NOFORM
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=true" %s | FileCheck %s --check-prefix=RETAIN

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
// CHECK:           onnx.Yield %[[ADD]] : tensor<2x2xf32>
// CHECK:         }) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:         %[[MIXED:.+]] = "onnx.Sub"(%[[RUNTIME]], %[[DEFERRED]])
// CHECK:         onnx.Return %[[DEFERRED]], %[[MIXED]]

// A later tensor constant cannot be added as a capture to an earlier expression.
// Build a new wrapper at the consumer and inline the operand expression instead.
func.func @later_capture_does_not_break_dominance() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "later_capture_weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %bias = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %result = "onnx.Add"(%transposed, %bias) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @later_capture_does_not_break_dominance
// CHECK-DAG:     %[[WEIGHT:.+]] = "onnx.InitializerSource"()
// CHECK-DAG:     %[[BIAS:.+]] = onnx.Constant dense<1.000000e+00>
// CHECK:         %[[DEFERRED:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[BIAS]]) ({
// CHECK:         ^bb0(%[[CAPTURED_WEIGHT:.+]]: tensor<2x2xf32>, %[[CAPTURED_BIAS:.+]]: tensor<2x2xf32>):
// CHECK:           %[[TRANSPOSE:.+]] = "onnx.Transpose"(%[[CAPTURED_WEIGHT]])
// CHECK:           %[[ADD:.+]] = "onnx.Add"(%[[TRANSPOSE]], %[[CAPTURED_BIAS]])
// CHECK:           onnx.Yield %[[ADD]] : tensor<2x2xf32>
// CHECK:         onnx.Return %[[DEFERRED]]

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

// Formation must not fold unrelated runtime operations. In particular, tensor
// name inference must observe both adjacent reshapes and the inner name.
func.func @preserve_unrelated_runtime_reshapes(%runtime: tensor<1x2x2xf32>) -> (tensor<1x1x4xf32>, tensor<2x2xf32>) {
  %inner = "tosa.reshape"(%runtime) {ResultNames = ["runtime_inner"], new_shape = array<i64: 1, 4>} : (tensor<1x2x2xf32>) -> tensor<1x4xf32>
  %outer = "tosa.reshape"(%inner) {new_shape = array<i64: 1, 1, 4>} : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "reshape_weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %outer, %transposed : tensor<1x1x4xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @preserve_unrelated_runtime_reshapes
// CHECK-SAME:  (%[[RUNTIME:.+]]: tensor<1x2x2xf32>)
// CHECK:         %[[INNER:.+]] = tosa.reshape %[[RUNTIME]] {ResultNames = ["runtime_inner"]
// CHECK-NEXT:    %[[OUTER:.+]] = tosa.reshape %[[INNER]]
// CHECK:         %[[DEFERRED:.+]] = "onnx.DeferredFoldable"
// CHECK:         onnx.Return %[[OUTER]], %[[DEFERRED]]

// NAMES-LABEL: func.func @preserve_unrelated_runtime_reshapes
// NAMES-SAME:  (%[[RUNTIME:.+]]: tensor<1x2x2xf32>)
// NAMES:         %[[INNER:.+]] = tosa.reshape %[[RUNTIME]] {ResultNames = ["runtime_inner"]
// NAMES-NEXT:    %[[OUTER:.+]] = tosa.reshape %[[INNER]] {new_shape = array<i64: 1, 1, 4>}
// NAMES:         %[[DEFERRED:.+]] = "onnx.DeferredFoldable"
// NAMES:         onnx.Return %[[OUTER]], %[[DEFERRED]]

// Initializer-only formation leaves ordinary dense expressions in their
// original order relative to runtime operations.
func.func @preserve_dense_expression_order(%runtime: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %constant = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %transposed = "onnx.Transpose"(%constant) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %marker = "onnx.Identity"(%runtime) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %sum = "onnx.Add"(%transposed, %constant) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %result = "onnx.Sub"(%marker, %sum) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @preserve_dense_expression_order
// CHECK:         %[[CONSTANT:.+]] = onnx.Constant
// CHECK-NEXT:    %[[TRANSPOSE:.+]] = "onnx.Transpose"(%[[CONSTANT]])
// CHECK-NEXT:    %[[MARKER:.+]] = "onnx.Identity"(%{{.+}})
// CHECK-NEXT:    %[[SUM:.+]] = "onnx.Add"(%[[TRANSPOSE]], %[[CONSTANT]])
// CHECK-NEXT:    %[[RESULT:.+]] = "onnx.Sub"(%[[MARKER]], %[[SUM]])
// CHECK-NEXT:    onnx.Return %[[RESULT]]

// A pure multi-result operation is retained as one single-result expression per
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
// CHECK:           onnx.Yield %[[SPLIT0]]#0
// CHECK:         %[[PART1:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[SPLITS]]) ({
// CHECK:           %[[SPLIT1:.+]]:3 = "onnx.Split"
// CHECK:           onnx.Yield %[[SPLIT1]]#1
// CHECK:         %[[PART2:.+]] = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[SPLITS]]) ({
// CHECK:           %[[SPLIT2:.+]]:3 = "onnx.Split"
// CHECK:           onnx.Yield %[[SPLIT2]]#2
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

// Generic cleanup is not the expression evaluation mechanism, but a
// semantics-preserving canonicalization may eliminate an identity terminal.
// Once the expression directly yields a capture, it is redundant and is removed.
func.func @canonicalized_identity_recipe() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%captured: tensor<2x2xf32>):
    %identity = "onnx.Identity"(%captured) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.Yield %identity : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>
}

// CANON-LABEL: func.func @canonicalized_identity_recipe
// CANON:         %[[SOURCE:.+]] = "onnx.InitializerSource"()
// CANON-NOT:     "onnx.DeferredFoldable"
// CANON-NOT:     "onnx.Identity"
// CANON:         onnx.Return %[[SOURCE]]

// Constprop treats a retained expression as opaque. Dense work inside the expression
// is evaluated only when the expression is explicitly materialized.
func.func @constprop_does_not_enter_recipe() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%captured: tensor<2x2xf32>):
    %one = onnx.Constant dense<1.0> : tensor<2x2xf32>
    %two = onnx.Constant dense<2.0> : tensor<2x2xf32>
    %three = "onnx.Add"(%one, %two) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %sum = "onnx.Add"(%captured, %three) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.Yield %sum : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>
}

// NOFORM-LABEL: func.func @constprop_does_not_enter_recipe
// NOFORM:         "onnx.DeferredFoldable"
// NOFORM:           %[[ONE:.+]] = onnx.Constant dense<1.000000e+00>
// NOFORM:           %[[TWO:.+]] = onnx.Constant dense<2.000000e+00>
// NOFORM:           %[[THREE:.+]] = "onnx.Add"(%[[ONE]], %[[TWO]])
// NOFORM:           %[[SUM:.+]] = "onnx.Add"(%{{.+}}, %[[THREE]])
// NOFORM:           onnx.Yield %[[SUM]]

// A retained expression remains valid when later dialect conversion lowers an
// operation in its body.
func.func @lowered_recipe_operation() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %reshaped = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 2>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.Yield %reshaped : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// Compiler-owned expressions created by downstream normalization remain opaque to
// constant propagation; available captures do not request their evaluation.
func.func @retain_compiler_owned_recipe(%runtime: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %constant = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %recipe = "onnx.DeferredFoldable"(%constant) ({
  ^bb0(%captured: tensor<2x2xf32>):
    %transposed = "onnx.Transpose"(%captured) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.Yield %transposed : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %result = "onnx.Add"(%runtime, %recipe) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>
}

// RETAIN-LABEL: func.func @retain_compiler_owned_recipe
// RETAIN: %[[CONSTANT:.+]] = onnx.Constant
// RETAIN: %[[RECIPE:.+]] = "onnx.DeferredFoldable"(%[[CONSTANT]])
// RETAIN: %[[TRANSPOSE:.+]] = "onnx.Transpose"
// RETAIN: onnx.Yield %[[TRANSPOSE]]
// RETAIN: "onnx.Add"(%{{.+}}, %[[RECIPE]])
