// RUN: onnx-mlir-opt --xfe-onnx-opset-verifier %s -split-input-file -verify-diagnostics

// -----
// Static Flatten is in the default deny-list.
// RUN 2 (empty list): no errors at all, tool exits 0.
func.func @static_flatten_fails(%arg0: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  // expected-error@+1 {{'onnx.Flatten' op disallowed in XFE ONNX opset}}
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  onnx.Return %0 : tensor<6x4xf32>
}

// -----
// Dynamic primary operand: static gate skips the op.
func.func @dynamic_flatten_passes(%arg0: tensor<?x3x4xf32>) -> tensor<?x4xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<?x3x4xf32>) -> tensor<?x4xf32>
  onnx.Return %0 : tensor<?x4xf32>
}

// -----
// Dynamic axes operand (tensor<?xi64>): static gate skips Squeeze because
// not all operands are fully-static.
func.func @squeeze_dynamic_axes_passes(%arg0: tensor<2x1x4xf32>,
                                        %axes: tensor<?xi64>) -> tensor<2x4xf32> {
  %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<2x1x4xf32>, tensor<?xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
}

// -----
// Both Flatten and Squeeze are in the default deny-list; collect-all means
// both errors are reported before the single signalPassFailure().
func.func @collect_all_violations(%arg0: tensor<2x3x4xf32>,
                                   %arg1: tensor<2x1x4xf32>)
    -> (tensor<6x4xf32>, tensor<2x4xf32>) {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  // expected-error@+1 {{'onnx.Flatten' op disallowed in XFE ONNX opset}}
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  // expected-error@+1 {{'onnx.Squeeze' op disallowed in XFE ONNX opset}}
  %1 = "onnx.Squeeze"(%arg1, %axes) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  onnx.Return %0, %1 : tensor<6x4xf32>, tensor<2x4xf32>
}
