// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

func.func @test_pad_const_pads(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[0, 2, 2, 4]> : tensor<4xi64>
  %1 = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  %cst = "onnx.NoValue"() {value} : () -> none
  %2 = "onnx.Pad"(%arg0, %0, %1, %cst) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_pad_const_pads
  // CHECK-SAME:     ([[VAR_arg0:%.+]]: tensor<16x13xf32>) -> tensor<18x19xf32> {
  // CHECK: [[VAR_0:%.+]] = onnx.Constant dense<[0, 2, 2, 4]> : tensor<4xi64>
  // CHECK: [[VAR_1:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK: [[VAR_2:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[VAR_3:%.+]] = "onnx.Pad"([[VAR_arg0]], [[VAR_0]], [[VAR_1]], [[VAR_2]]) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<18x19xf32>
}

// -----

func.func @test_pad_const_pad_axes(%arg0: tensor<1x3x4x5xf32>) -> tensor<?x?x?x?xf32> {
  %0 = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %1 = onnx.Constant dense<[0, 3, 0, 4]> : tensor<4xi64>
  %2 = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  %3 = "onnx.Pad"(%arg0, %1, %2, %0) {mode = "constant"}: (tensor<1x3x4x5xf32>, tensor<4xi64>, tensor<1xf32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
  return %3 : tensor<?x?x?x?xf32>

  // CHECK-LABEL: func @test_pad_const_pad_axes
  // CHECK-SAME: (%[[VAR_arg0:.*]]: tensor<1x3x4x5xf32>) -> tensor<?x3x?x12xf32> {
  // CHECK: %[[CONST_0:.*]] = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  // CHECK: %[[CONST_1:.*]] = onnx.Constant dense<[0, 3, 0, 4]> : tensor<4xi64>
  // CHECK: %[[CONST_2:.*]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  // CHECK: %[[PAD_0:.*]] = "onnx.Pad"(%[[VAR_arg0]], %[[CONST_1]], %[[CONST_2]], %[[CONST_0]]) {mode = "constant"} : (tensor<1x3x4x5xf32>, tensor<4xi64>, tensor<1xf32>, tensor<2xi64>) -> tensor<?x3x?x12xf32>
  // %3 = "onnx.Pad"(%arg0, %1, %2, %0)  // CHECK: return %[[PAD_0]] : tensor<?x3x?x12xf32>
}

// -----

func.func @test_pad_const_axes(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
    %0 = onnx.Constant dense<[1, 3]> : tensor<2xi64>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
    %2 = "onnx.Pad"(%arg0, %arg1, %1, %0) {mode = "constant"} : (tensor<1x2x3x4xf32>, tensor<4xi64>, tensor<1xf32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
    return %2 : tensor<?x?x?x?xf32>

    // CHECK-LABEL: func @test_pad_const_axes
    // CHECK-SAME: (%[[VAR_arg0:.*]]: tensor<1x2x3x4xf32>, %[[VAR_arg1:.*]]: tensor<4xi64>) -> tensor<?x?x?x?xf32> {
    // CHECK: %[[CONST_0:.*]] = onnx.Constant dense<[1, 3]> : tensor<2xi64>
    // CHECK: %[[CONST_1:.*]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
    // CHECK: %[[PAD_0:.*]] = "onnx.Pad"(%[[VAR_arg0]], %[[VAR_arg1]], %[[CONST_1]], %[[CONST_0]]) {mode = "constant"} : (tensor<1x2x3x4xf32>, tensor<4xi64>, tensor<1xf32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
}

// -----

func.func @test_pad_all_dynamic(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<4xi64>, %arg2: tensor<f32>, %arg3: tensor<2xi64>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %arg3) {mode = "constant"} : (tensor<1x3x4x5xf32>, tensor<4xi64>, tensor<f32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>

  // CHECK-LABEL: func @test_pad_all_dynamic
  // CHECK-SAME: ([[VAR_arg0:%.+]]: tensor<1x3x4x5xf32>, [[VAR_arg1:%.+]]: tensor<4xi64>, [[VAR_arg2:%.+]]: tensor<f32>, [[VAR_arg3:%.+]]: tensor<2xi64>) -> tensor<?x?x?x?xf32> {
  // CHECK: [[VAR_0:%.+]] = "onnx.Pad"([[VAR_arg0]], [[VAR_arg1]], [[VAR_arg2]], [[VAR_arg3]]) {mode = "constant"} : (tensor<1x3x4x5xf32>, tensor<4xi64>, tensor<f32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
  // CHECK: return [[VAR_0]] : tensor<?x?x?x?xf32>
}