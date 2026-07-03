// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

func.func @test_reducesumv11_axes(%arg0: tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32> {
  %0 = "onnx.ReduceSumV11"(%arg0) {axes = [1], keepdims = 0 : si64} : (tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32>
  onnx.Return %0 : tensor<1x512x640xf32>
// CHECK-LABEL:  func.func @test_reducesumv11_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x512x640xf32>) -> tensor<1x512x640xf32> {
// CHECK-DAG:       [[AXES_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[REDUCED_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[AXES_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x512x640xf32>, tensor<1xi64>) -> tensor<1x512x640xf32>
// CHECK:           onnx.Return [[REDUCED_]] : tensor<1x512x640xf32>
// CHECK:         }
}

// -----

func.func @test_reducesumv11_noaxes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %0 = "onnx.ReduceSumV11"(%arg0) {keepdims = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<1x1x1xf32>
  onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL:  func.func @test_reducesumv11_noaxes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<1x1x1xf32>
// CHECK:         }
}
