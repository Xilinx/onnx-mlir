// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=true" %s --split-input-file | FileCheck %s

func.func @reassociate_adds(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "c0", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %c1 = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "c1", offset = 16 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %0 = "onnx.Add"(%c0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "onnx.Add"(%c1, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  onnx.Return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @reassociate_adds
// CHECK-DAG: [[C0:%.+]] = "onnx.InitializerSource"() {{.*}}name = "c0"
// CHECK-DAG: [[C1:%.+]] = "onnx.InitializerSource"() {{.*}}name = "c1"
// CHECK: [[DEFERRED_SUM:%.+]] = "onnx.DeferredFoldable"([[C0]], [[C1]]) ({
// CHECK:   "onnx.Add"
// CHECK:   onnx.DeferredYield
// CHECK: })
// CHECK: [[RESULT:%.+]] = "onnx.Add"(%arg0, [[DEFERRED_SUM]])
// CHECK: onnx.Return [[RESULT]]

// -----

// Merely containing an initializer is insufficient: the complete expression
// must have no runtime dependency.
func.func @do_not_defer_runtime_dependent_expression(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %bias = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "bias", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %mixed = "onnx.Add"(%bias, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %result = "onnx.Sub"(%arg0, %mixed) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  onnx.Return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @do_not_defer_runtime_dependent_expression
// CHECK: [[BIAS:%.+]] = "onnx.InitializerSource"() {{.*}}name = "bias"
// CHECK: [[MIXED:%.+]] = "onnx.Add"(%arg1, [[BIAS]])
// CHECK: [[RESULT:%.+]] = "onnx.Sub"(%arg0, [[MIXED]])
// CHECK-NOT: "onnx.Neg"([[MIXED]])
// CHECK: onnx.Return [[RESULT]]

// -----

// Verify that an expression created by one algebraic rewrite remains a
// constant expression for a later rewrite iteration.
func.func @reassociate_created_deferred_expression(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %bias = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "bias", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %offset = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Sub"(%arg0, %bias) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "onnx.Add"(%0, %offset) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  onnx.Return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @reassociate_created_deferred_expression
// CHECK-DAG: [[BIAS:%.+]] = "onnx.InitializerSource"() {{.*}}name = "bias"
// CHECK-DAG: [[OFFSET:%.+]] = onnx.Constant dense<5.000000e-01>
// CHECK: [[DEFERRED_SUM:%.+]] = "onnx.DeferredFoldable"([[BIAS]], [[OFFSET]]) ({
// CHECK:   [[NEGATED:%.+]] = "onnx.Neg"
// CHECK:   "onnx.Add"([[NEGATED]], {{%.+}})
// CHECK:   onnx.DeferredYield
// CHECK: })
// CHECK: [[RESULT:%.+]] = "onnx.Add"(%arg0, [[DEFERRED_SUM]])
// CHECK: onnx.Return [[RESULT]]

// -----

func.func @reassociate_transformed_initializer(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %offset = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %0 = "onnx.Add"(%arg0, %transposed) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = "onnx.Add"(%0, %offset) : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
  onnx.Return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @reassociate_transformed_initializer
// CHECK-DAG: [[WEIGHT:%.+]] = "onnx.InitializerSource"() {{.*}}name = "weight"
// CHECK-DAG: [[OFFSET:%.+]] = onnx.Constant dense<5.000000e-01>
// CHECK: [[DEFERRED_SUM:%.+]] = "onnx.DeferredFoldable"([[WEIGHT]], [[OFFSET]]) ({
// CHECK:   [[TRANSPOSED:%.+]] = "onnx.Transpose"
// CHECK:   "onnx.Add"([[TRANSPOSED]], {{%.+}})
// CHECK:   onnx.DeferredYield
// CHECK: })
// CHECK: [[RESULT:%.+]] = "onnx.Add"(%arg0, [[DEFERRED_SUM]])
// CHECK: onnx.Return [[RESULT]]

// -----

func.func @subtract_initializer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %bias = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "bias", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %0 = "onnx.Sub"(%arg0, %bias) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  onnx.Return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @subtract_initializer
// CHECK-DAG: [[BIAS:%.+]] = "onnx.InitializerSource"() {{.*}}name = "bias"
// CHECK: [[NEGATED:%.+]] = "onnx.DeferredFoldable"([[BIAS]]) ({
// CHECK:   "onnx.Neg"
// CHECK:   onnx.DeferredYield
// CHECK: })
// CHECK: [[RESULT:%.+]] = "onnx.Add"(%arg0, [[NEGATED]])
// CHECK: onnx.Return [[RESULT]]

// -----

func.func @reassociate_muls(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %weight) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "onnx.Mul"(%0, %scale) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  onnx.Return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @reassociate_muls
// CHECK-DAG: [[WEIGHT:%.+]] = "onnx.InitializerSource"() {{.*}}name = "weight"
// CHECK-DAG: [[SCALE:%.+]] = onnx.Constant dense<5.000000e-01>
// CHECK: [[DEFERRED_SCALE:%.+]] = "onnx.DeferredFoldable"([[WEIGHT]], [[SCALE]]) ({
// CHECK:   "onnx.Mul"
// CHECK:   onnx.DeferredYield
// CHECK: })
// CHECK: [[RESULT:%.+]] = "onnx.Mul"(%arg0, [[DEFERRED_SCALE]])
// CHECK: onnx.Return [[RESULT]]

// -----

func.func @distribute_mul(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %bias = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "bias", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %scale = onnx.Constant dense<4.0> : tensor<f32>
  %0 = "onnx.Add"(%bias, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "onnx.Mul"(%0, %scale) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  onnx.Return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @distribute_mul
// CHECK-DAG: [[BIAS:%.+]] = "onnx.InitializerSource"() {{.*}}name = "bias"
// CHECK-DAG: [[SCALE:%.+]] = onnx.Constant dense<4.000000e+00>
// CHECK: [[SCALED_INPUT:%.+]] = "onnx.Mul"(%arg0, [[SCALE]])
// CHECK: [[SCALED_BIAS:%.+]] = "onnx.DeferredFoldable"([[BIAS]], [[SCALE]]) ({
// CHECK:   "onnx.Mul"
// CHECK:   onnx.DeferredYield
// CHECK: })
// CHECK: [[RESULT:%.+]] = "onnx.Add"([[SCALED_INPUT]], [[SCALED_BIAS]])
// CHECK: onnx.Return [[RESULT]]

// -----

// A non-deterministic operation must not extend a deferred constant
// expression, even when all of its operands are constant-foldable.
func.func @do_not_reassociate_nondeterministic_expression(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %seed_data = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "seed_data", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<4xf32>
  %random = "onnx.RandomNormalLike"(%seed_data) {dtype = 1 : si64, mean = 0.0 : f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  %offset = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %random) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "onnx.Add"(%0, %offset) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  onnx.Return %1 : tensor<4xf32>
}

// CHECK-LABEL: func.func @do_not_reassociate_nondeterministic_expression
// CHECK-DAG: [[SEED_DATA:%.+]] = "onnx.InitializerSource"() {{.*}}name = "seed_data"
// CHECK-DAG: [[OFFSET:%.+]] = onnx.Constant dense<5.000000e-01>
// CHECK: [[RANDOM:%.+]] = "onnx.RandomNormalLike"([[SEED_DATA]])
// CHECK: [[SUM:%.+]] = "onnx.Add"(%arg0, [[RANDOM]])
// CHECK: [[RESULT:%.+]] = "onnx.Add"([[SUM]], [[OFFSET]])
// CHECK: onnx.Return [[RESULT]]
