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

// -----

// COM: Existing symmetric pads are kept on the begin side; only end pads grow.
// COM: H=W=7, pads=1, kernel=2, stride=2 -> ceil output=5, delta=1, end pads 1->2.
func.func @test_avgpool_ceil_existing_sym_pads(%arg0: tensor<1x3x7x7xf32>) -> tensor<1x3x5x5xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x5x5xf32>
  return %0 : tensor<1x3x5x5xf32>

  // CHECK-LABEL: test_avgpool_ceil_existing_sym_pads
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [1, 1, 2, 2]
  // CHECK-SAME: -> tensor<1x3x5x5xf32>
}

// -----

// COM: Begin-only pads [1, 2, 0, 0]: H is divisible (delta=0), W needs +1 end pad.
func.func @test_avgpool_ceil_existing_begin_pads(%arg0: tensor<1x3x7x7xf32>) -> tensor<1x3x4x5xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [1, 2, 0, 0], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x5xf32>
  return %0 : tensor<1x3x4x5xf32>

  // CHECK-LABEL: test_avgpool_ceil_existing_begin_pads
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [1, 2, 0, 1]
  // CHECK-SAME: -> tensor<1x3x4x5xf32>
}

// -----

// COM: Existing end pads [0, 0, 1, 2] grow only where ceil still overhangs (W +1).
func.func @test_avgpool_ceil_existing_end_pads(%arg0: tensor<1x3x7x7xf32>) -> tensor<1x3x4x5xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 1, 2], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x5xf32>
  return %0 : tensor<1x3x4x5xf32>

  // CHECK-LABEL: test_avgpool_ceil_existing_end_pads
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 3]
  // CHECK-SAME: -> tensor<1x3x4x5xf32>
}

// -----

// COM: Per-axis strides [2, 3] on 7x8: H delta=1, W exactly divisible.
func.func @test_avgpool_ceil_uneven_strides(%arg0: tensor<1x3x7x8xf32>) -> tensor<1x3x4x3xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 3]} : (tensor<1x3x7x8xf32>) -> tensor<1x3x4x3xf32>
  return %0 : tensor<1x3x4x3xf32>

  // CHECK-LABEL: test_avgpool_ceil_uneven_strides
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 0]
  // CHECK-SAME: strides = [2, 3]
  // CHECK-SAME: -> tensor<1x3x4x3xf32>
}

// -----

// COM: Larger kernel than stride, mixed kernel/stride: kernel=[3,3], strides=[3,2] on 10x11.
// COM: H ceil output=4 delta=2; W divisible so pads stay 0.
func.func @test_avgpool_ceil_kernel_gt_stride(%arg0: tensor<1x3x10x11xf32>) -> tensor<1x3x4x5xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [3, 2]} : (tensor<1x3x10x11xf32>) -> tensor<1x3x4x5xf32>
  return %0 : tensor<1x3x4x5xf32>

  // CHECK-LABEL: test_avgpool_ceil_kernel_gt_stride
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 2, 0]
  // CHECK-SAME: strides = [3, 2]
  // CHECK-SAME: -> tensor<1x3x4x5xf32>
}

// -----

// COM: Non-unit dilations: effective kernel=(2-1)*2+1=3 on 8x8, stride=2 -> delta=1.
func.func @test_avgpool_ceil_dilations(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, dilations = [2, 2], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x8x8xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_dilations
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: dilations = [2, 2]
  // CHECK-SAME: pads = [0, 0, 1, 1]
  // CHECK-SAME: -> tensor<1x3x4x4xf32>
}

// -----

// COM: Combined: existing pads + mixed kernel/stride. kernel=[5,3], strides=[3,2],
// COM: pads=[2,1,1,0] on 13x17 -> H end 1->2, W end 0->1, output 5x9.
func.func @test_avgpool_ceil_pads_kernel_stride(%arg0: tensor<1x3x13x17xf32>) -> tensor<1x3x5x9xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [5, 3], pads = [2, 1, 1, 0], strides = [3, 2]} : (tensor<1x3x13x17xf32>) -> tensor<1x3x5x9xf32>
  return %0 : tensor<1x3x5x9xf32>

  // CHECK-LABEL: test_avgpool_ceil_pads_kernel_stride
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [2, 1, 2, 1]
  // CHECK-SAME: strides = [3, 2]
  // CHECK-SAME: -> tensor<1x3x5x9xf32>
}

// -----

// COM: 1D AveragePool (N,C,L): L=7, kernel=2, stride=2 -> ceil output=4, pads [0, 1].
func.func @test_avgpool_ceil_1d(%arg0: tensor<1x3x7xf32>) -> tensor<1x3x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2], pads = [0, 0], strides = [2]} : (tensor<1x3x7xf32>) -> tensor<1x3x4xf32>
  return %0 : tensor<1x3x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_1d
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 1]
  // CHECK-SAME: -> tensor<1x3x4xf32>
}

// -----

// COM: 3D AveragePool (N,C,D,H,W): 7x8x9, kernel/stride 2 -> pads [D_b,H_b,W_b, D_e,H_e,W_e]
// COM: D delta=1, H divisible, W delta=1 -> [0,0,0, 1,0,1], output 4x4x5.
func.func @test_avgpool_ceil_3d(%arg0: tensor<1x1x7x8x9xf32>) -> tensor<1x1x4x4x5xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2, 2], pads = [0, 0, 0, 0, 0, 0], strides = [2, 2, 2]} : (tensor<1x1x7x8x9xf32>) -> tensor<1x1x4x4x5xf32>
  return %0 : tensor<1x1x4x4x5xf32>

  // CHECK-LABEL: test_avgpool_ceil_3d
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 0, 1, 0, 1]
  // CHECK-SAME: -> tensor<1x1x4x4x5xf32>
}

// -----

// COM: Missing pads attribute defaults to 0; still folded and pads are materialized.
func.func @test_avgpool_ceil_omitted_pads(%arg0: tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_omitted_pads
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [0, 0, 1, 1]
  // CHECK-SAME: -> tensor<1x3x4x4xf32>
}

// -----

// COM: Non-uniform begin pad from the shape-inference suite: H=30 pads_begin=1,
// COM: W=32 pads=0, kernel=2, stride=2 -> ceil 16x16, only H end grows 0->1.
func.func @test_avgpool_ceil_nonunif_begin_pad(%arg0: tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  return %0 : tensor<5x5x16x16xf32>

  // CHECK-LABEL: test_avgpool_ceil_nonunif_begin_pad
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: ceil_mode = 0 : si64
  // CHECK-SAME: pads = [1, 0, 1, 0]
  // CHECK-SAME: -> tensor<5x5x16x16xf32>
}

// -----

// COM: auto_pad SAME_UPPER derives output independently of ceil_mode; do not fold.
func.func @test_avgpool_ceil_auto_pad_same_upper(%arg0: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 1 : si64, count_include_pad = 0 : si64, kernel_shape = [4, 4], strides = [4, 4]} : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  return %0 : tensor<5x5x4x4xf32>

  // CHECK-LABEL: test_avgpool_ceil_auto_pad_same_upper
  // CHECK: "onnx.AveragePool"(%arg0)
  // CHECK-SAME: auto_pad = "SAME_UPPER"
  // CHECK-SAME: ceil_mode = 1 : si64
}
