// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --set-onnx-node-name --split-input-file %s | FileCheck %s

module { 
  func.func @set_for_multiple_ops(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x2xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
    %1 = "onnx.Relu"(%0) : (tensor<3x2xf32>) -> tensor<3x2xf32>
    onnx.Return %1 : tensor<3x2xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()

// CHECK-LABEL:  func.func @set_for_multiple_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<3x2xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) {onnx_node_name = "onnx.Add_0"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {onnx_node_name = "onnx.Relu_1"} : (tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<3x2xf32>
// CHECK:         }
// CHECK:         "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

func.func @user_the_previous_onnx_node_name(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "residual/add1"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  %1 = "onnx.Relu"(%0) : (tensor<3x2xf32>) -> tensor<3x2xf32>
  onnx.Return %1 : tensor<3x2xf32>

// CHECK-LABEL:  func.func @user_the_previous_onnx_node_name
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<3x2xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) {onnx_node_name = "residual/add1"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {onnx_node_name = "residual/add1_onnx.Relu_0"} : (tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<3x2xf32>
// CHECK:         }
}

// -----

func.func @duplicated_onnx_node_name(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "residual/add1"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  %1 = "onnx.Add"(%0, %arg1) {onnx_node_name = "residual/add1"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  onnx.Return %1 : tensor<3x2xf32>

// CHECK-LABEL:  func.func @duplicated_onnx_node_name
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>, [[PARAM_1_:%.+]]: tensor<3x2xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) {onnx_node_name = "residual/add1"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_1_]]) {onnx_node_name = "residual/add1_0"} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<3x2xf32>
// CHECK:         }
}

// -----

func.func @skip_constant_foldable_expression(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %weight = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "weight", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2x2xf32>
  %transposed = "onnx.Transpose"(%weight) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %result = "onnx.Add"(%arg0, %transposed) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  onnx.Return %result : tensor<2x2xf32>

// CHECK-LABEL:  func.func @skip_constant_foldable_expression
// CHECK:           [[WEIGHT:%.+]] = "onnx.InitializerSource"() {{.*}} : () -> tensor<2x2xf32>
// CHECK-NOT:       onnx_node_name
// CHECK:           [[TRANSPOSED:%.+]] = "onnx.Transpose"([[WEIGHT]]) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NOT:       onnx_node_name
// CHECK:           [[RESULT:%.+]] = "onnx.Add"(%arg0, [[TRANSPOSED]]) {onnx_node_name = "onnx.Add_0"}
// CHECK:           onnx.Return [[RESULT]]
}

