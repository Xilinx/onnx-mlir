// RUN: onnx-mlir-opt --verify-xfe-dialect="disallowed-ops=Squeeze" %s -split-input-file -verify-diagnostics

// -----
// With disallowed-ops=Squeeze, static Flatten is allowed.
func.func @flatten_allowed_with_squeeze_override(%arg0: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  onnx.Return %0 : tensor<6x4xf32>
}

// -----
// With disallowed-ops=Squeeze, static Squeeze is rejected.
func.func @squeeze_rejected_with_override(%arg0: tensor<2x1x4xf32>) -> tensor<2x4xf32> {
  %axes = onnx.Constant dense<[1]> : tensor<1xi64>
  // expected-error@+1 {{'onnx.Squeeze' op disallowed in XFE dialect}}
  %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<2x1x4xf32>, tensor<1xi64>) -> tensor<2x4xf32>
  onnx.Return %0 : tensor<2x4xf32>
}
