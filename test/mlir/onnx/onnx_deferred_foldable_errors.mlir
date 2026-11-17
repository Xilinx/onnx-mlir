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

func.func @capture_count_mismatch() -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op region control flow edge from parent operands to Region #0: source has 1 operands, but target successor <to region #0 with 2 inputs> needs 2}}
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>, %unused: tensor<2x2xf32>):
    %0 = "onnx.Identity"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %0 : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// -----

func.func @capture_type_mismatch() -> tensor<2x2xi32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op along control flow edge from parent operands to Region #0: source type #0 'tensor<2x2xf32>' should match input type #0 'tensor<2x2xi32>'}}
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xi32>):
    %0 = "onnx.Identity"(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    onnx.DeferredYield %0 : tensor<2x2xi32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}

// -----

func.func @yield_type_mismatch() -> tensor<2x2xi32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  // expected-error@+1 {{'onnx.DeferredFoldable' op along control flow edge from Operation onnx.DeferredYield to parent results: source type #0 'tensor<2x2xf32>' should match input type #0 'tensor<2x2xi32>'}}
  %result = "onnx.DeferredFoldable"(%weight) ({
  ^bb0(%arg0: tensor<2x2xf32>):
    %0 = "onnx.Identity"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    onnx.DeferredYield %0 : tensor<2x2xf32>
  }) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
