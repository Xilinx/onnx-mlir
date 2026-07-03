// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Canonicalization: onnx.AveragePool ceil_mode=1 -> ceil_mode=0 + padding.
/// ONNX AveragePool is channel-first: X is [N, C, H, W]; pads are
/// [H_begin, W_begin, H_end, W_end].
//===----------------------------------------------------------------------===//

// COM: Non-divisible spatial dims -> ceil_mode dropped, end pads grown by delta.
// COM: H=W=7, kernel=2, stride=2 -> ceil output=4, delta=1 per axis.
func.func @test_avgpool_ceil_to_pad(%arg0: tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_to_pad
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 1]
  // CHECK-SAME: -> tensor<1x3x4x4xf32>
}

// -----

// COM: Exactly-divisible spatial dims -> ceil_mode dropped, pads unchanged.
func.func @test_avgpool_ceil_divisible(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_divisible
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 0, 0]
  // CHECK-SAME: -> tensor<1x3x4x4xf32>
}

// -----

// COM: count_include_pad=1 -> not folded (would change the divisor).
func.func @test_avgpool_ceil_count_include_pad(%arg0: tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_count_include_pad
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 1 : si64
}

// -----

// COM: Dynamic spatial dim -> cannot fold, op left unchanged (ceil_mode=1).
func.func @test_avgpool_ceil_dynamic(%arg0: tensor<1x3x?x7xf32>) -> tensor<1x3x?x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x?x7xf32>) -> tensor<1x3x?x4xf32>
  return %0 : tensor<1x3x?x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_dynamic
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 1 : si64
}

// -----

// COM: Asymmetric H/W: only H needs a delta (H=7 -> +1), W is divisible (W=8).
func.func @test_avgpool_ceil_asymmetric(%arg0: tensor<1x3x7x8xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x7x8xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_asymmetric
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 0]
  // CHECK-SAME: -> tensor<1x3x4x4xf32>
}
