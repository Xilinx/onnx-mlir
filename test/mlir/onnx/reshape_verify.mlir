// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @multiple_inferred_dims(%x: tensor<?x?xf32>) -> tensor<1x?x?xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, -1, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
  // expected-error @+1 {{shape must contain at most one -1 dimension}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<1x?x?xf32>
  return %y : tensor<1x?x?xf32>
}

// -----

func.func @multiple_inferred_dims_tosa(%x: tensor<?x?xf32>) -> tensor<1x?x?xf32> {
  %shape = "tosa.const"() <{value = dense<[1, -1, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
  // expected-error @+1 {{shape must contain at most one -1 dimension}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<1x?x?xf32>
  return %y : tensor<1x?x?xf32>
}

// -----

func.func @multiple_inferred_dims_arith(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %shape = arith.constant dense<[-1, -1]> : tensor<2xi64>
  // expected-error @+1 {{shape must contain at most one -1 dimension}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

func.func @invalid_negative_dim(%x: tensor<6xf32>) -> tensor<?x3xf32> {
  %shape = "onnx.Constant"() {value = dense<[-2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{shape dimensions must be greater than or equal to -1}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<6xf32>, tensor<2xi64>) -> tensor<?x3xf32>
  return %y : tensor<?x3xf32>
}

// -----

func.func @literal_zero_and_inference(%x: tensor<0x?xf32>) -> tensor<1x0x?xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, 0, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
  // expected-error @+1 {{Allowzero is set and shape contains both -1 and 0}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 1 : si64} : (tensor<0x?xf32>, tensor<3xi64>) -> tensor<1x0x?xf32>
  return %y : tensor<1x0x?xf32>
}

// -----

func.func @literal_zero_and_inference_tosa(%x: tensor<0x?xf32>) -> tensor<1x0x?xf32> {
  %shape = "tosa.const"() <{value = dense<[1, 0, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
  // expected-error @+1 {{Allowzero is set and shape contains both -1 and 0}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 1 : si64} : (tensor<0x?xf32>, tensor<3xi64>) -> tensor<1x0x?xf32>
  return %y : tensor<1x0x?xf32>
}

// -----

func.func @zero_copy_out_of_rank(%x: tensor<6xf32>) -> tensor<1x6xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{0-dim cannot copy dimension beyond rank of input}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<6xf32>, tensor<2xi64>) -> tensor<1x6xf32>
  return %y : tensor<1x6xf32>
}

// -----

func.func @invalid_allowzero(%x: tensor<6xf32>) -> tensor<6xf32> {
  %shape = "onnx.Constant"() {value = dense<[6]> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{allowzero must be 0 or 1}}
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 2 : si64} : (tensor<6xf32>, tensor<1xi64>) -> tensor<6xf32>
  return %y : tensor<6xf32>
}

// -----

func.func @single_inferred_dim(%x: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, -1, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<?x3xf32>, tensor<3xi64>) -> tensor<1x?x3xf32>
  return %y : tensor<1x?x3xf32>
}

// -----

func.func @zero_copy_and_inference(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %shape = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  return %y : tensor<2x3xf32>
}

// -----

func.func @literal_zeros(%x: tensor<0x3xf32>) -> tensor<1x0x3xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, 0, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 1 : si64} : (tensor<0x3xf32>, tensor<3xi64>) -> tensor<1x0x3xf32>
  return %y : tensor<1x0x3xf32>
}

// -----

func.func @scalar(%x: tensor<1xf32>) -> tensor<f32> {
  %shape = "onnx.Constant"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<1xf32>, tensor<0xi64>) -> tensor<f32>
  return %y : tensor<f32>
}

// -----

func.func @zero_copy_unranked_input(%x: tensor<*xf32>) -> tensor<?x3xf32> {
  %shape = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<2xi64>) -> tensor<?x3xf32>
  return %y : tensor<?x3xf32>
}

// -----

func.func @single_inferred_dim_tosa(%x: tensor<?x3xf32>) -> tensor<1x?x3xf32> {
  %shape = "tosa.const"() <{value = dense<[1, -1, 3]> : tensor<3xi64>}> : () -> tensor<3xi64>
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<?x3xf32>, tensor<3xi64>) -> tensor<1x?x3xf32>
  return %y : tensor<1x?x3xf32>
}

// -----

func.func @runtime_shape(%x: tensor<?x?xf32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %y = "onnx.Reshape"(%x, %shape) {allowzero = 1 : si64} : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %y : tensor<?x?x?xf32>
}
