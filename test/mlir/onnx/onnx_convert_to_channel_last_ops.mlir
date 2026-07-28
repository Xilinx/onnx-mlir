// RUN: onnx-mlir-opt --convert-to-channel-last --shape-inference %s | FileCheck %s
// RUN: onnx-mlir-opt --convert-to-channel-last="whitelist=onnx.Conv" --shape-inference %s | FileCheck %s --check-prefix=CHECK-CONV
// RUN: onnx-mlir-opt --convert-to-channel-last="whitelist=onnx.Conv,onnx.AveragePool" --shape-inference %s | FileCheck %s --check-prefix=CHECK-BOTH
// RUN: onnx-mlir-opt --convert-to-channel-last="whitelist=onnx.NotAnOp" %s -verify-diagnostics

//===----------------------------------------------------------------------===//
/// The `whitelist` option restricts the conversion to the named source ops. An
/// empty list converts every supported op.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_ops_option
// CHECK-CONV-LABEL: func.func @test_ops_option
// CHECK-BOTH-LABEL: func.func @test_ops_option
// expected-error@+1 {{convert-to-channel-last: unsupported op 'onnx.NotAnOp'}}
func.func @test_ops_option(
    %arg0: tensor<1x3x28x28xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: tensor<64xf32>,
    %arg3: tensor<1x64x28x28xf32>) -> (tensor<1x64x26x26xf32>, tensor<1x64x14x14xf32>) {
  %conv = "onnx.Conv"(%arg0, %arg1, %arg2) {
    dilations = [1, 1],
    group = 1 : si64,
    pads = [0, 0, 0, 0],
    strides = [1, 1]
  } : (tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x26x26xf32>
  %pool = "onnx.AveragePool"(%arg3) {
    kernel_shape = [2, 2],
    strides = [2, 2],
    pads = [0, 0, 0, 0]
  } : (tensor<1x64x28x28xf32>) -> tensor<1x64x14x14xf32>
  onnx.Return %conv, %pool : tensor<1x64x26x26xf32>, tensor<1x64x14x14xf32>

  // CHECK: "onnx.XFEConv"
  // CHECK: "onnx.XFEAveragePool"

  // CHECK-CONV: "onnx.XFEConv"
  // CHECK-CONV-NOT: "onnx.XFEAveragePool"
  // CHECK-CONV: "onnx.AveragePool"

  // CHECK-BOTH: "onnx.XFEConv"
  // CHECK-BOTH: "onnx.XFEAveragePool"
}
