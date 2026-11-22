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

// A multi-result producer is retained once, preserving its shared work.
func.func @retain_multi_result_expression() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 48 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x6xf32>
  %splits = onnx.Constant dense<2> : tensor<3xi64>
  %parts:3 = "onnx.Split"(%weight, %splits) {axis = 1 : si64} : (tensor<2x6xf32>, tensor<3xi64>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
  onnx.Return %parts#0, %parts#1, %parts#2 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
}

// CHECK-LABEL: func.func @retain_multi_result_expression
// CHECK-DAG:     %[[WEIGHT:.+]] = "onnx.InitializerSource"()
// CHECK-DAG:     %[[SPLITS:.+]] = onnx.Constant dense<2>
// CHECK:         %[[PARTS:.+]]:3 = "onnx.DeferredFoldable"(%[[WEIGHT]], %[[SPLITS]]) ({
// CHECK:           %[[SPLIT:.+]]:3 = "onnx.Split"
// CHECK:           onnx.Yield %[[SPLIT]]#0, %[[SPLIT]]#1, %[[SPLIT]]#2
// CHECK:         onnx.Return %[[PARTS]]#0, %[[PARTS]]#1, %[[PARTS]]#2

// -----

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

// Standalone formation preserves unrelated dead operations both in an eager
// function and alongside a deferred chain that exceeds ten operations.
func.func @no_formation_roots() -> tensor<2xi64> {
  %live = onnx.Constant dense<[4, 8]> : tensor<2xi64>
  %dead = onnx.Constant dense<[4, 8]> : tensor<2xi64>
  %dead_add = "onnx.Add"(%live, %live) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %sum = "onnx.Add"(%live, %live) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %unused = "onnx.Neg"(%sum) : (tensor<2xi64>) -> tensor<2xi64>
  onnx.Return %live : tensor<2xi64>
}
// CHECK-LABEL: func.func @no_formation_roots
// CHECK-NEXT: %[[LIVE:.+]] = onnx.Constant dense<[4, 8]> : tensor<2xi64>
// CHECK-NEXT: %{{.+}} = onnx.Constant dense<[4, 8]> : tensor<2xi64>
// CHECK-NEXT: %{{.+}} = "onnx.Add"(%[[LIVE]], %[[LIVE]])
// CHECK-NEXT: %[[SUM:.+]] = "onnx.Add"(%[[LIVE]], %[[LIVE]])
// CHECK-NEXT: %{{.+}} = "onnx.Neg"(%[[SUM]])
// CHECK-NEXT: onnx.Return %[[LIVE]]

func.func @mixed_deep_chain(%arg: tensor<2xf32>) -> tensor<2xf32> {
  %dead = onnx.Constant dense<19.0> : tensor<2xf32>
  %unrelated = "onnx.Neg"(%arg) : (tensor<2xf32>) -> tensor<2xf32>
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "unread.onnx", scope = []} : () -> tensor<2xf32>
  %mixed = "onnx.Add"(%source, %arg) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %v1 = "onnx.Neg"(%source) : (tensor<2xf32>) -> tensor<2xf32>
  %v2 = "onnx.Neg"(%v1) : (tensor<2xf32>) -> tensor<2xf32>
  %v3 = "onnx.Neg"(%v2) : (tensor<2xf32>) -> tensor<2xf32>
  %v4 = "onnx.Neg"(%v3) : (tensor<2xf32>) -> tensor<2xf32>
  %v5 = "onnx.Neg"(%v4) : (tensor<2xf32>) -> tensor<2xf32>
  %v6 = "onnx.Neg"(%v5) : (tensor<2xf32>) -> tensor<2xf32>
  %v7 = "onnx.Neg"(%v6) : (tensor<2xf32>) -> tensor<2xf32>
  %v8 = "onnx.Neg"(%v7) : (tensor<2xf32>) -> tensor<2xf32>
  %v9 = "onnx.Neg"(%v8) : (tensor<2xf32>) -> tensor<2xf32>
  %v10 = "onnx.Neg"(%v9) : (tensor<2xf32>) -> tensor<2xf32>
  %v11 = "onnx.Neg"(%v10) : (tensor<2xf32>) -> tensor<2xf32>
  %v12 = "onnx.Neg"(%v11) : (tensor<2xf32>) -> tensor<2xf32>
  %discard = "onnx.Neg"(%v12) : (tensor<2xf32>) -> tensor<2xf32>
  onnx.Return %v12 : tensor<2xf32>
}
// CHECK-LABEL: func.func @mixed_deep_chain
// CHECK-SAME: (%[[ARG:.+]]: tensor<2xf32>)
// CHECK-NEXT: %{{.+}} = onnx.Constant dense<1.900000e+01> : tensor<2xf32>
// CHECK-NEXT: %{{.+}} = "onnx.Neg"(%[[ARG]])
// CHECK-NEXT: %[[SOURCE:.+]] = "onnx.InitializerSource"
// CHECK-NEXT: %{{.+}} = "onnx.Add"(%[[SOURCE]], %[[ARG]])
// CHECK-NEXT: %[[RECIPE:.+]] = "onnx.DeferredFoldable"(%[[SOURCE]])
// CHECK: ^bb0(
// CHECK-COUNT-12: = "onnx.Neg"
// CHECK-NEXT: onnx.Yield
// CHECK-NEXT: })
// CHECK-NEXT: onnx.Return %[[RECIPE]]

// An ordinary region owner cannot form a deferred expression, but operations nested in
// its blocks can. A source defined outside the branch remains a valid capture.
func.func @nested_formation(%condition: tensor<i1>) -> tensor<2xf32> {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "nested", path = "unread.onnx", scope = []} : () -> tensor<2xf32>
  %result = "onnx.If"(%condition) ({
    %neg = "onnx.Neg"(%source) : (tensor<2xf32>) -> tensor<2xf32>
    onnx.Yield %neg : tensor<2xf32>
  }, {
    onnx.Yield %source : tensor<2xf32>
  }) : (tensor<i1>) -> tensor<2xf32>
  onnx.Return %result : tensor<2xf32>
}
// CHECK-LABEL: func.func @nested_formation
// CHECK: %[[S:.+]] = "onnx.InitializerSource"
// CHECK: %[[IF:.+]] = "onnx.If"
// CHECK-NEXT: %[[R:.+]] = "onnx.DeferredFoldable"(%[[S]])
// CHECK-NEXT: ^bb0(%[[A:.+]]: tensor<2xf32>):
// CHECK-NEXT: %[[N:.+]] = "onnx.Neg"(%[[A]])
// CHECK-NEXT: onnx.Yield %[[N]]
// CHECK-NEXT: })
// CHECK-NEXT: onnx.Yield %[[R]]
// CHECK: onnx.Yield %[[S]]
// CHECK: onnx.Return %[[IF]]
