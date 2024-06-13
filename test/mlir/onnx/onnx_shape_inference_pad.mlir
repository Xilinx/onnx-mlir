// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

func.func @test_pad_const_pad_axes(%arg0: tensor<1x3x4x5xf32>) -> tensor<?x?x?x?xf32> {
    %0 = onnx.Constant dense<[1, 3]> : tensor<2xi64>
    %1 = onnx.Constant dense<[0, 3, 0, 4]> : tensor<4xi64>
    %2 = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
    %3 = "onnx.Pad"(%arg0, %1, %2, %0) {mode = "constant", onnx_node_name = "onnx.Pad_0"} : (tensor<1x3x4x5xf32>, tensor<4xi64>, tensor<1xf32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
    return %3 : tensor<?x?x?x?xf32>

    // CHECK-LABEL: func @test_pad_const_pad_axes
    // CHECK: %[[CONST_0:.*]] = onnx.Constant dense<[1, 3]> : tensor<2xi64>
    // CHECK: %[[CONST_1:.*]] = onnx.Constant dense<[0, 3, 0, 4]> : tensor<4xi64>
    // CHECK: %[[CONST_2:.*]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
    // CHECK: %[[PAD_0:.*]] = "onnx.Pad"(%arg0, %[[CONST_1]], %[[CONST_2]], %[[CONST_0]]) {mode = "constant", onnx_node_name = "onnx.Pad_0"} : (tensor<1x3x4x5xf32>, tensor<4xi64>, tensor<1xf32>, tensor<2xi64>) -> tensor<?x3x?x12xf32>
    // CHECK: return %[[PAD_0]] : tensor<?x3x?x12xf32>
}