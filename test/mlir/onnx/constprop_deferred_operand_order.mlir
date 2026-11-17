// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=false" %s | FileCheck %s
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=true" %s | FileCheck %s
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=false" %s | FileCheck %s --check-prefix=RHS

func.func @source_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xf32>
  %add = "onnx.Add"(%source, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%source, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @source_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[SOURCE:.*]] = "onnx.InitializerSource"
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[RUNTIME]], %[[SOURCE]])
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[RUNTIME]], %[[SOURCE]])
// CHECK: return %[[ADD]], %[[MUL]]

func.func @deferred_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xf32>
  %recipe = "onnx.DeferredFoldable"(%source) ({
  ^bb0(%data: tensor<2xf32>):
    %negative = "onnx.Neg"(%data) : (tensor<2xf32>) -> tensor<2xf32>
    onnx.DeferredYield %negative : tensor<2xf32>
  }) : (tensor<2xf32>) -> tensor<2xf32>
  %add = "onnx.Add"(%recipe, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%recipe, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @deferred_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[RECIPE:.*]] = "onnx.DeferredFoldable"
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[RUNTIME]], %[[RECIPE]])
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[RUNTIME]], %[[RECIPE]])
// CHECK: return %[[ADD]], %[[MUL]]

func.func @literal_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %literal = onnx.Constant dense<2.0> : tensor<2xf32>
  %add = "onnx.Add"(%literal, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%literal, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @literal_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[LITERAL:.*]] = onnx.Constant dense<2.000000e+00> : tensor<2xf32>
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[RUNTIME]], %[[LITERAL]])
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[RUNTIME]], %[[LITERAL]])
// CHECK: return %[[ADD]], %[[MUL]]

func.func @dequantized_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xi8>
  %recipe = "onnx.DeferredFoldable"(%source) ({
  ^bb0(%data: tensor<2xi8>):
    %scale = onnx.Constant dense<0.125> : tensor<f32>
    %zero = onnx.Constant dense<0> : tensor<i8>
    %dq = "onnx.DequantizeLinear"(%data, %scale, %zero) {axis = 0 : si64} : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
    onnx.DeferredYield %dq : tensor<2xf32>
  }) : (tensor<2xi8>) -> tensor<2xf32>
  %add = "onnx.Add"(%recipe, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%recipe, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @dequantized_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[RECIPE:.*]] = "onnx.DeferredFoldable"
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[RECIPE]], %[[RUNTIME]])
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[RECIPE]], %[[RUNTIME]])
// CHECK: return %[[ADD]], %[[MUL]]

func.func @literal_before_dequantized_operand() -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xi8>
  %literal = onnx.Constant dense<2.0> : tensor<2xf32>
  %recipe = "onnx.DeferredFoldable"(%source) ({
  ^bb0(%data: tensor<2xi8>):
    %scale = onnx.Constant dense<0.125> : tensor<f32>
    %zero = onnx.Constant dense<0> : tensor<i8>
    %dq = "onnx.DequantizeLinear"(%data, %scale, %zero) {axis = 0 : si64} : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
    onnx.DeferredYield %dq : tensor<2xf32>
  }) : (tensor<2xi8>) -> tensor<2xf32>
  %add = "onnx.Add"(%literal, %recipe) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%literal, %recipe) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// RHS-LABEL: func.func @literal_before_dequantized_operand
// RHS: %[[LITERAL:.*]] = onnx.Constant dense<2.000000e+00> : tensor<2xf32>
// RHS: %[[RECIPE:.*]] = "onnx.DeferredFoldable"
// RHS: %[[ADD:.*]] = "onnx.Add"(%[[RECIPE]], %[[LITERAL]])
// RHS: %[[MUL:.*]] = "onnx.Mul"(%[[RECIPE]], %[[LITERAL]])
// RHS: return %[[ADD]], %[[MUL]]
