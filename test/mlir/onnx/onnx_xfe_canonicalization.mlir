// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Canonicalization tests for XFE Operations
/// Domain: com.amd.xfe
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
/// XFEAveragePool ceil_mode -> padding folding.
/// Channel-last layout: X is [N, H, W, C]; pads are [H_begin, W_begin, H_end,
/// W_end].
//===----------------------------------------------------------------------===//

// COM: Non-divisible spatial dims: ceil_mode dropped, end pads grown by delta.
// COM: H=W=7, kernel=2, stride=2 -> num=5, ceil output=4, delta=1 per axis.
func.func @test_xfe_avgpool_ceil_to_pad(%arg0: tensor<1x7x7x3xf32>) -> tensor<1x4x4x3xf32> {
  %0 = "onnx.XFEAveragePool"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x7x7x3xf32>) -> tensor<1x4x4x3xf32>
  onnx.Return %0 : tensor<1x4x4x3xf32>

  // CHECK-LABEL: test_xfe_avgpool_ceil_to_pad
  // CHECK: "onnx.XFEAveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 1]
  // CHECK-SAME: -> tensor<1x4x4x3xf32>
}

// -----

// COM: Exactly-divisible spatial dims: ceil_mode dropped, pads unchanged.
// COM: H=W=8, kernel=2, stride=2 -> num=6, divisible, delta=0.
func.func @test_xfe_avgpool_ceil_divisible(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x4x4x3xf32> {
  %0 = "onnx.XFEAveragePool"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x8x8x3xf32>) -> tensor<1x4x4x3xf32>
  onnx.Return %0 : tensor<1x4x4x3xf32>

  // CHECK-LABEL: test_xfe_avgpool_ceil_divisible
  // CHECK: "onnx.XFEAveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 0, 0]
  // CHECK-SAME: -> tensor<1x4x4x3xf32>
}

// -----

// COM: Dynamic spatial dim: cannot fold, op is left unchanged (ceil_mode=1).
func.func @test_xfe_avgpool_ceil_dynamic(%arg0: tensor<1x?x7x3xf32>) -> tensor<1x?x4x3xf32> {
  %0 = "onnx.XFEAveragePool"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x?x7x3xf32>) -> tensor<1x?x4x3xf32>
  onnx.Return %0 : tensor<1x?x4x3xf32>

  // CHECK-LABEL: test_xfe_avgpool_ceil_dynamic
  // CHECK: "onnx.XFEAveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 1 : si64
}

// -----

// COM: Asymmetric H/W: only H needs a delta (H=7 -> +1), W is divisible (W=8).
func.func @test_xfe_avgpool_ceil_asymmetric(%arg0: tensor<1x7x8x3xf32>) -> tensor<1x4x4x3xf32> {
  %0 = "onnx.XFEAveragePool"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x7x8x3xf32>) -> tensor<1x4x4x3xf32>
  onnx.Return %0 : tensor<1x4x4x3xf32>

  // CHECK-LABEL: test_xfe_avgpool_ceil_asymmetric
  // CHECK: "onnx.XFEAveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 0]
  // CHECK-SAME: -> tensor<1x4x4x3xf32>
}
