// RUN: onnx-mlir-opt --xfe-onnx-opset-verifier="disallowed-ops=" %s -split-input-file -verify-diagnostics

// -----

func.func @negative_axis_attr_fails(%arg0: tensor<2x3xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x7xf32> {
  // expected-error@+1 {{'onnx.Concat' op negative axis value is disallowed in XFE ONNX opset}}
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  onnx.Return %0 : tensor<2x7xf32>
}

// -----

func.func @negative_axes_attr_fails(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
  // expected-error@+1 {{'onnx.ReduceMeanV13' op negative axis value is disallowed in XFE ONNX opset}}
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1, 1], keepdims = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<2x1x1xf32>
  onnx.Return %0 : tensor<2x1x1xf32>
}

// -----

func.func @negative_axes_operand_fails(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
  %axes = onnx.Constant dense<[-1, 1]> : tensor<2xi64>
  // expected-error@+1 {{'onnx.ReduceMean' op negative axis value is disallowed in XFE ONNX opset}}
  %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
  onnx.Return %0 : tensor<2x1x1xf32>
}

// -----

func.func @positive_axes_pass(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
  onnx.Return %0 : tensor<2x1x1xf32>
}

// -----

func.func @unranked_input_negative_axis_passes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}


// -----

func.func @dynamic_axes_operand_passes(%arg0: tensor<2x3x4xf32>, %axes: tensor<2xi64>) -> tensor<2x1x1xf32> {
  %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
  onnx.Return %0 : tensor<2x1x1xf32>
}
