// RUN: onnx-mlir-opt --xfe-onnx-opset-verifier="disallowed-ops= verify-non-negative-axis=false" %s -split-input-file

func.func @negative_axis_allowed_when_disabled(%arg0: tensor<2x3xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x7xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  onnx.Return %0 : tensor<2x7xf32>
}
