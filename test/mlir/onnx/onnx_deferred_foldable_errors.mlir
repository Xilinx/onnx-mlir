// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --split-input-file --verify-diagnostics %s

func.func @nested_wrapper() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %inner = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %0 : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op must inline a captured onnx.DeferredFoldable body}}
  %outer = "onnx.DeferredFoldable"(%inner) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %0 = "onnx.Identity"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %0 : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %outer : tensor<2x2xf32>
}

// -----

func.func @nonterminal_yield() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op terminal operation must immediately precede its yield}}
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %0 = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = "onnx.Identity"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %0 : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// -----

func.func @runtime_capture(%runtime: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error@+1 {{'onnx.DeferredFoldable' op capture must be constant or deferred constant}}
  %result = "onnx.DeferredFoldable"(%runtime) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %0 = "onnx.Identity"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %0 : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// -----

func.func @dead_recipe_operation() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op body operation must contribute to the yielded recipe: onnx.Identity}}
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %dead = "onnx.Identity"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %terminal = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %terminal : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// -----

func.func @non_onnx_recipe_operation() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op body operation must belong to the ONNX dialect: tosa.reshape}}
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %reshaped = "tosa.reshape"(%arg0) {new_shape = array<i64: 2, 2>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %reshaped : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
