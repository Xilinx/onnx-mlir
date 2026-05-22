// RUN: onnx-mlir-opt --split-input-file --transfer-qdq-pixel-unshuffle-chain-to-conv2d %s | FileCheck %s

// =============================================================================
// Positive case 1 - exact PSV-style 6-op chain (B = 4) with a trailing
// NCHW->NHWC layout transpose.  The pass must fold the 5 chain ops (reshape ->
// transpose([0,2,1,3]) -> reshape -> transpose([0,3,1,2]) -> reshape) into a
// stride-4 identity-weight onnx.Conv that produces the NCHW SpaceToDepth
// tensor.  The trailing NCHW->NHWC transpose remains and is now applied to the
// Conv result.
// =============================================================================
func.func @qdq_pixel_unshuffle_chain_psv_b4(
    %x: tensor<1x3x256x256x!quant.uniform<u8:f32, 5.000000e-01:5>>)
    -> tensor<1x64x64x48x!quant.uniform<u8:f32, 5.000000e-01:5>> {
  %s0 = onnx.Constant dense<[3, 64, 4, 256]>   : tensor<4xi64>
  %s1 = onnx.Constant dense<[12, 64, 64, 4]>   : tensor<4xi64>
  %s2 = onnx.Constant dense<[1, 48, 64, 64]>   : tensor<4xi64>

  %r0 = "onnx.Reshape"(%x,  %s0) {allowzero = 0 : si64}
      : (tensor<1x3x256x256x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4xi64>)
      -> tensor<3x64x4x256x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %t0 = "onnx.Transpose"(%r0) {perm = [0, 2, 1, 3]}
      : (tensor<3x64x4x256x!quant.uniform<u8:f32, 5.000000e-01:5>>)
      -> tensor<3x4x64x256x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %r1 = "onnx.Reshape"(%t0, %s1) {allowzero = 0 : si64}
      : (tensor<3x4x64x256x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4xi64>)
      -> tensor<12x64x64x4x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %t1 = "onnx.Transpose"(%r1) {perm = [0, 3, 1, 2]}
      : (tensor<12x64x64x4x!quant.uniform<u8:f32, 5.000000e-01:5>>)
      -> tensor<12x4x64x64x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %r2 = "onnx.Reshape"(%t1, %s2) {allowzero = 0 : si64}
      : (tensor<12x4x64x64x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4xi64>)
      -> tensor<1x48x64x64x!quant.uniform<u8:f32, 5.000000e-01:5>>
  // Trailing NCHW->NHWC layout switch, exactly like PSV_ver_3.4.0 does.
  %y  = "onnx.Transpose"(%r2) {perm = [0, 2, 3, 1]}
      : (tensor<1x48x64x64x!quant.uniform<u8:f32, 5.000000e-01:5>>)
      -> tensor<1x64x64x48x!quant.uniform<u8:f32, 5.000000e-01:5>>
  return %y : tensor<1x64x64x48x!quant.uniform<u8:f32, 5.000000e-01:5>>
}

// CHECK-LABEL: func.func @qdq_pixel_unshuffle_chain_psv_b4

// Identity weights: shape = [C*B*B, C, B, B] = [48, 3, 4, 4]
// CHECK: onnx.Constant
// CHECK-SAME: tensor<48x3x4x4xi8>
// CHECK-SAME: tensor<48x3x4x4x!quant.uniform<i8:f32, 1.000000e+00>>

// Zero bias: shape = [C*B*B] = [48], i32 storage (matches QLinearConv spec
// and the XIR convention used by every other quantized Conv in the lowering).
// CHECK: onnx.Constant
// CHECK-SAME: dense<0> : tensor<48xi32>
// CHECK-SAME: tensor<48x!quant.uniform<i32:f32, 5.000000e-01>>

// CHECK: %[[CONV:.*]] = "onnx.Conv"
// CHECK-SAME: auto_pad = "NOTSET"
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: group = 1
// CHECK-SAME: kernel_shape = [4, 4]
// CHECK-SAME: pads = [0, 0, 0, 0]
// CHECK-SAME: strides = [4, 4]
// CHECK-SAME: -> tensor<1x48x64x64x!quant.uniform<u8:f32, 5.000000e-01:5>>

// Trailing layout-switch transpose stays in place and now consumes the Conv.
// CHECK: "onnx.Transpose"(%[[CONV]]) {perm = [0, 2, 3, 1]}

// CHECK-NOT: onnx.Reshape

// -----

// =============================================================================
// Positive case 2 - same algebra at B = 2 on a smaller tensor.  Here we keep
// the previous "trailing identity transpose" form so the test also exercises
// the case where a downstream transpose is essentially a no-op; it should
// remain in the IR and consume the Conv result directly.
// =============================================================================
func.func @qdq_pixel_unshuffle_chain_b2(
    %x: tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>)
    -> tensor<1x12x4x4x!quant.uniform<u8:f32, 5.000000e-01:5>> {
  %s0 = onnx.Constant dense<[1, 3, 4, 16]>     : tensor<4xi64>
  %s1 = onnx.Constant dense<[1, 12, 2, 8]>     : tensor<4xi64>
  %s2 = onnx.Constant dense<[1, 12, 4, 4]>     : tensor<4xi64>

  %r0 = "onnx.Reshape"(%x,  %s0) {allowzero = 0 : si64}
      : (tensor<1x3x8x8x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4xi64>)
      -> tensor<1x3x4x16x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %t0 = "onnx.Transpose"(%r0) {perm = [0, 2, 1, 3]}
      : (tensor<1x3x4x16x!quant.uniform<u8:f32, 5.000000e-01:5>>)
      -> tensor<1x4x3x16x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %r1 = "onnx.Reshape"(%t0, %s1) {allowzero = 0 : si64}
      : (tensor<1x4x3x16x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4xi64>)
      -> tensor<1x12x2x8x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %t1 = "onnx.Transpose"(%r1) {perm = [0, 3, 1, 2]}
      : (tensor<1x12x2x8x!quant.uniform<u8:f32, 5.000000e-01:5>>)
      -> tensor<1x8x12x2x!quant.uniform<u8:f32, 5.000000e-01:5>>
  %y  = "onnx.Reshape"(%t1, %s2) {allowzero = 0 : si64}
      : (tensor<1x8x12x2x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<4xi64>)
      -> tensor<1x12x4x4x!quant.uniform<u8:f32, 5.000000e-01:5>>
  return %y : tensor<1x12x4x4x!quant.uniform<u8:f32, 5.000000e-01:5>>
}

// CHECK-LABEL: func.func @qdq_pixel_unshuffle_chain_b2

// CHECK: onnx.Constant
// CHECK-SAME: tensor<12x3x2x2xi8>
// CHECK-SAME: tensor<12x3x2x2x!quant.uniform<i8:f32, 1.000000e+00>>

// CHECK: onnx.Constant
// CHECK-SAME: dense<0> : tensor<12xi32>
// CHECK-SAME: tensor<12x!quant.uniform<i32:f32, 5.000000e-01>>

// CHECK: "onnx.Conv"
// CHECK-SAME: auto_pad = "NOTSET"
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: group = 1
// CHECK-SAME: kernel_shape = [2, 2]
// CHECK-SAME: pads = [0, 0, 0, 0]
// CHECK-SAME: strides = [2, 2]
// CHECK-SAME: -> tensor<1x12x4x4x!quant.uniform<u8:f32, 5.000000e-01:5>>

// CHECK-NOT: onnx.Reshape

// -----

// =============================================================================
// Negative case - wrong perm on t0; pattern must not match and the chain
// must remain intact.
// =============================================================================
func.func @bad_perm_t0(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x12x4x4xf32> {
  %s0 = onnx.Constant dense<[1, 3, 4, 16]> : tensor<4xi64>
  %s1 = onnx.Constant dense<[1, 12, 2, 8]> : tensor<4xi64>
  %s2 = onnx.Constant dense<[1, 12, 4, 4]> : tensor<4xi64>
  %r0 = "onnx.Reshape"(%x,  %s0) {allowzero = 0 : si64}  : (tensor<1x3x8x8xf32>, tensor<4xi64>) -> tensor<1x3x4x16xf32>
  // perm is [0,2,3,1] instead of the required [0,2,1,3]
  %t0 = "onnx.Transpose"(%r0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x16xf32>) -> tensor<1x4x16x3xf32>
  %r1 = "onnx.Reshape"(%t0, %s1) {allowzero = 0 : si64} : (tensor<1x4x16x3xf32>, tensor<4xi64>) -> tensor<1x12x2x8xf32>
  %t1 = "onnx.Transpose"(%r1) {perm = [0, 3, 1, 2]} : (tensor<1x12x2x8xf32>) -> tensor<1x8x12x2xf32>
  %y  = "onnx.Reshape"(%t1, %s2) {allowzero = 0 : si64} : (tensor<1x8x12x2xf32>, tensor<4xi64>) -> tensor<1x12x4x4xf32>
  return %y : tensor<1x12x4x4xf32>
}

// CHECK-LABEL: func.func @bad_perm_t0
// CHECK: onnx.Reshape
// CHECK: onnx.Transpose
// CHECK-NOT: onnx.Conv
