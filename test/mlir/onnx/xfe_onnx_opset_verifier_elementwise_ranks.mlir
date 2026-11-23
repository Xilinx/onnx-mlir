// RUN: onnx-mlir-opt %s --xfe-onnx-opset-verifier
// RUN: onnx-mlir-opt --xfe-onnx-opset-verifier="verify-equal-elementwise-operand-ranks=true" -verify-diagnostics %s

func.func @mixed_rank(%x: tensor<2x3xf32>, %y: tensor<3xf32>) -> tensor<2x3xf32> {
  // expected-error@+1 {{'onnx.Add' op expected equal ranks for elementwise broadcast operands, but operand #0 has rank 2 and operand #1 has rank 1}}
  %add = "onnx.Add"(%x, %y) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %add : tensor<2x3xf32>
}

func.func @mixed_dynamic_ranks(%x: tensor<?x3xf32>, %y: tensor<3xf32>) -> tensor<?x3xf32> {
  // expected-error@+1 {{'onnx.Add' op expected equal ranks for elementwise broadcast operands, but operand #0 has rank 2 and operand #1 has rank 1}}
  %add = "onnx.Add"(%x, %y) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
  return %add : tensor<?x3xf32>
}
