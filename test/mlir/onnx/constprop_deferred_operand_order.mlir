// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates

// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=false" %s | FileCheck %s
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=true" %s | FileCheck %s
// RUN: onnx-mlir-opt --constprop-onnx="form-deferred-foldable-expressions=false" %s | FileCheck %s --check-prefix=RHS
// RUN: onnx-mlir-opt --constprop-onnx --onnx-const-prop-disable-pattern=AddConstCommutative1 %s | FileCheck %s --check-prefixes=DISABLE,NO-ADD
// RUN: onnx-mlir-opt --constprop-onnx --onnx-const-prop-disable-pattern=MulConstCommutative1 %s | FileCheck %s --check-prefixes=DISABLE,NO-MUL

func.func @source_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xf32>
  %add = "onnx.Add"(%source, %runtime) {onnx_node_name = "source_Add"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%source, %runtime) {onnx_node_name = "source_Mul"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @source_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[SOURCE:.*]] = "onnx.InitializerSource"
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[RUNTIME]], %[[SOURCE]]) {onnx_node_name = "source_Add"}
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[RUNTIME]], %[[SOURCE]]) {onnx_node_name = "source_Mul"}
// CHECK: return %[[ADD]], %[[MUL]]

// DISABLE-LABEL: func.func @source_operand_order(
// DISABLE-SAME: %[[RUNTIME:[^: ]+]]:
// DISABLE: %[[SOURCE:.*]] = "onnx.InitializerSource"
// NO-ADD: "onnx.Add"(%[[SOURCE]], %[[RUNTIME]])
// NO-ADD: "onnx.Mul"(%[[RUNTIME]], %[[SOURCE]])
// NO-MUL: "onnx.Add"(%[[RUNTIME]], %[[SOURCE]])
// NO-MUL: "onnx.Mul"(%[[SOURCE]], %[[RUNTIME]])

func.func @deferred_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xf32>
  %recipe = "onnx.DeferredFoldable"(%source) ({
  ^bb0(%data: tensor<2xf32>):
    %negative = "onnx.Neg"(%data) : (tensor<2xf32>) -> tensor<2xf32>
    onnx.Yield %negative : tensor<2xf32>
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
  %add = "onnx.Add"(%literal, %runtime) {onnx_node_name = "literal_Add"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%literal, %runtime) {onnx_node_name = "literal_Mul"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @literal_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[LITERAL:.*]] = onnx.Constant dense<2.000000e+00> : tensor<2xf32>
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[RUNTIME]], %[[LITERAL]]) {onnx_node_name = "literal_Add"}
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[RUNTIME]], %[[LITERAL]]) {onnx_node_name = "literal_Mul"}
// CHECK: return %[[ADD]], %[[MUL]]

func.func @dequantized_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "weight", path = "/not/read.onnx", scope = []} : () -> tensor<2xi8>
  %recipe = "onnx.DeferredFoldable"(%source) ({
  ^bb0(%data: tensor<2xi8>):
    %scale = onnx.Constant dense<0.125> : tensor<f32>
    %zero = onnx.Constant dense<0> : tensor<i8>
    %dq = "onnx.DequantizeLinear"(%data, %scale, %zero) {axis = 0 : si64} : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
    onnx.Yield %dq : tensor<2xf32>
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
    onnx.Yield %dq : tensor<2xf32>
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

// An explicit DQ of literal weights is not a literal constant for the
// pre-existing Add/Mul commutation rules, even though it can be evaluated.
func.func @explicit_dequantized_operand_order(%runtime: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %data = onnx.Constant dense<8> : tensor<2xi8>
  %scale = onnx.Constant dense<0.125> : tensor<f32>
  %zero = onnx.Constant dense<0> : tensor<i8>
  %dq = "onnx.DequantizeLinear"(%data, %scale, %zero) {axis = 0 : si64} : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
  %add = "onnx.Add"(%dq, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %mul = "onnx.Mul"(%dq, %runtime) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %add, %mul : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @explicit_dequantized_operand_order(
// CHECK-SAME: %[[RUNTIME:[^: ]+]]:
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"
// CHECK: %[[ADD:.*]] = "onnx.Add"(%[[DQ]], %[[RUNTIME]])
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[DQ]], %[[RUNTIME]])
// CHECK: return %[[ADD]], %[[MUL]]
