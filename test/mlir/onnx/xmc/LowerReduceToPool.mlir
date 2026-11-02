// RUN: onnx-mlir-opt --split-input-file --lower-reduce-to-pool %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// Layout: NCHW [batch, channels, height, width]
//
// NOTE: Several quantized cases below intentionally use *different* input vs.
// output quantization scale / zero-point. The original ReduceMean / ReduceSum
// op carries the canonical post-reduction quantization metadata, and the
// lowered AveragePool result must inherit the *output* scale / zero-point
// (not the input). The CHECK lines verify the lowered AveragePool op carries
// the output scale (e.g. `0.1:1`) rather than the input scale (e.g. `0.05:0`).

//===----------------------------------------------------------------------===//
// ReduceMean → AveragePool Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_spatial_hw
// NCHW: tensor<1x3x4x4> - N=1, C=3, H=4, W=4
// Reduce axes [2, 3] (H, W) -> output tensor<1x3x1x1>
// Full-spatial reduction with keepdims=true is emitted as ONNXGlobalAveragePool
// to preserve the natural [H, W] kernel downstream (matches legacy xmodel flow).
func.func @reduce_mean_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.1:1>>
    return %1 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.GlobalAveragePool"
// CHECK-SAME: !quant.uniform<i8:f32, 1.000000e-01:1>>
// CHECK-NOT: onnx.ReduceMean
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_mean_single_axis_h
// NCHW: tensor<1x3x8x4> - reduce axis [2] (H); single-axis mean is not lowered
// (aligned with xcompiler QDQ mean template: axes.size() == 2).
func.func @reduce_mean_single_axis_h(%arg0: tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.1:1>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_mean_single_axis_w
// NCHW: tensor<1x3x4x8> - reduce axis [3] (W); not lowered (axes.size() == 2).
func.func @reduce_mean_single_axis_w(%arg0: tensor<1x3x4x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x4x1x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = onnx.Constant dense<[3]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x4x1x!quant.uniform<i8:f32, 0.1:1>>
    return %1 : tensor<1x3x4x1x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_mean_negative_axis
// NCHW: tensor<1x3x4x4> - reduce axis [-2] = axis 2 (H); not lowered.
func.func @reduce_mean_negative_axis(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = onnx.Constant dense<[-2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.1:1>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// -----

//===----------------------------------------------------------------------===//
// ReduceSum → AveragePool + Mul Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_sum_spatial_hw
// NCHW: tensor<1x3x4x4> - reduce axes [2, 3] (H, W)
// The ReduceSum -> AveragePool + Mul(count) lowering was removed; the op
// is left as-is so the DPU's native qlinear_reduction_sum kernel handles it.
func.func @reduce_sum_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.8:2>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.8:2>>
    return %1 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.8:2>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: "onnx.AveragePool"
// CHECK-NOT: "onnx.Mul"

// -----

// CHECK-LABEL: @reduce_sum_single_axis
// NCHW: tensor<1x3x8x4> - reduce axis [2] (H)
// Single-axis spatial ReduceSum is similarly left as-is.
func.func @reduce_sum_single_axis(%arg0: tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.4:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.4:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.4:0>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: "onnx.AveragePool"
// CHECK-NOT: "onnx.Mul"

// -----

//===----------------------------------------------------------------------===//
// ReduceMax → MaxPool (Spatial) Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_max_spatial_hw
// NCHW: tensor<1x3x4x4> - reduce axes [2, 3] (H, W)
func.func @reduce_max_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.MaxPoolSingleOut"(%arg0)
// CHECK-SAME: kernel_shape = [4, 4]
// CHECK-SAME: strides = [4, 4]
// CHECK-SAME: (tensor<1x3x4x4x{{.*}}>) -> tensor<1x3x1x1x
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.ReduceMax

// -----

// CHECK-LABEL: @reduce_max_single_spatial_axis
// NCHW: tensor<1x3x8x4> - reduce axis [2] (H)
func.func @reduce_max_single_spatial_axis(%arg0: tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.MaxPoolSingleOut"(%arg0)
// CHECK-SAME: kernel_shape = [8, 1]
// CHECK-SAME: strides = [8, 1]
// CHECK-SAME: (tensor<1x3x8x4x{{.*}}>) -> tensor<1x3x1x4x
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.ReduceMax

// -----

// CHECK-LABEL: @reduce_max_h_square_extent_with_trailing_w
// NCHW: tensor<1x16x16x160> - reduce axis [2] (H). H = 16 is a perfect square,
// but W follows the reduced axis, so the kernel must stay [16, 1] on the
// original shape. Squaring it to [4, 4] would reshape 1x16x16x160 to 1x16x4x4.
func.func @reduce_max_h_square_extent_with_trailing_w(%arg0: tensor<1x16x16x160x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x16x160x!quant.uniform<u8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>>
    return %1 : tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.MaxPoolSingleOut"(%arg0)
// CHECK-SAME: kernel_shape = [16, 1]
// CHECK-SAME: strides = [16, 1]
// CHECK-SAME: (tensor<1x16x16x160x{{.*}}>) -> tensor<1x16x1x160x
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.ReduceMax

// -----

// CHECK-LABEL: @reduce_max_h_small_square_extent_with_trailing_w
// NCHW: tensor<1x16x4x160> - reduce axis [2] (H = 4, a perfect square).
func.func @reduce_max_h_small_square_extent_with_trailing_w(%arg0: tensor<1x16x4x160x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x4x160x!quant.uniform<u8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>>
    return %1 : tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.MaxPoolSingleOut"(%arg0)
// CHECK-SAME: kernel_shape = [4, 1]
// CHECK-SAME: (tensor<1x16x4x160x{{.*}}>) -> tensor<1x16x1x160x
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.ReduceMax

// -----

// CHECK-LABEL: @reduce_max_h_exceeds_kernel_limit_not_pooled
// NCHW: tensor<1x16x96x160> - reduce axis [2] (H = 96 > MAX_KERNEL_SIZE) with a
// trailing W. Flattening to 1x16x1x15360 with kernel [1, 96] would pool 96
// consecutive W elements instead of a column, so the ReduceMax is kept.
func.func @reduce_max_h_exceeds_kernel_limit_not_pooled(%arg0: tensor<1x16x96x160x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x96x160x!quant.uniform<u8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>>
    return %1 : tensor<1x16x1x160x!quant.uniform<u8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMax"
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.MaxPoolSingleOut

// -----

// CHECK-LABEL: @reduce_max_lifted_rank3_channel
// A rank-3 [N, C, L] ReduceMax over C reaches this pass as [1, N, C, L] over
// axis 2 (ReplaceQDQReductionPass pads rank < 4 with leading 1s).
func.func @reduce_max_lifted_rank3_channel(%arg0: tensor<1x1x16x160x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x1x1x160x!quant.uniform<u8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x1x16x160x!quant.uniform<u8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x1x160x!quant.uniform<u8:f32, 0.05:0>>
    return %1 : tensor<1x1x1x160x!quant.uniform<u8:f32, 0.05:0>>
}
// CHECK-NOT: onnx.Reshape
// CHECK: "onnx.MaxPoolSingleOut"(%arg0)
// CHECK-SAME: kernel_shape = [16, 1]
// CHECK-SAME: (tensor<1x1x16x160x{{.*}}>) -> tensor<1x1x1x160x
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.ReduceMax

// -----

//===----------------------------------------------------------------------===//
// ReduceMax → MaxPool (Channel via Reshape) Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_max_channel_axis
// NCHW: tensor<2x8x1x1> - reduce axis [1] (C). Every dim after C is 1, so the
// reduced elements are innermost and the reshape trick applies.
func.func @reduce_max_channel_axis(%arg0: tensor<2x8x1x1x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<2x1x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<2x8x1x1x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<2x1x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<2x1x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.Reshape"(%arg0
// CHECK-SAME: -> tensor<1x1x2x8x
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK-SAME: kernel_shape = [1, 8]
// CHECK-SAME: (tensor<1x1x2x8x{{.*}}>) -> tensor<1x1x2x1x
// CHECK: "onnx.Reshape"
// CHECK-SAME: -> tensor<2x1x1x1x
// CHECK-NOT: onnx.ReduceMax

// -----

// CHECK-LABEL: @reduce_max_channel_axis_with_spatial_not_pooled
// NCHW: tensor<1x8x4x4> - reduce axis [1] (C). H and W follow C, so reshaping
// to [1, 1, H*W, C] would pool consecutive W elements of one channel.
func.func @reduce_max_channel_axis_with_spatial_not_pooled(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMax"
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.MaxPoolSingleOut

// -----

// CHECK-LABEL: @reduce_max_channel_axis_large_spatial_not_pooled
// NCHW: tensor<1x16x96x160> - reduce axis [1] (C = 16). Previously reshaped to
// 1x1x15360x16 and emitted a pool whose declared result type was transposed.
func.func @reduce_max_channel_axis_large_spatial_not_pooled(%arg0: tensor<1x16x96x160x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x1x96x160x!quant.uniform<u8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x96x160x!quant.uniform<u8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x96x160x!quant.uniform<u8:f32, 0.05:0>>
    return %1 : tensor<1x1x96x160x!quant.uniform<u8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMax"
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.MaxPoolSingleOut

// -----

// CHECK-LABEL: @reduce_max_rank3_channel_axis_not_pooled
// Rank-3 [N, C, L] ReduceMax over C (L > 1 follows C): kept as a reduction.
func.func @reduce_max_rank3_channel_axis_not_pooled(%arg0: tensor<1x16x160x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x1x160x!quant.uniform<u8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x16x160x!quant.uniform<u8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x160x!quant.uniform<u8:f32, 0.05:0>>
    return %1 : tensor<1x1x160x!quant.uniform<u8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMax"
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.MaxPoolSingleOut

// -----

//===----------------------------------------------------------------------===//
// ReduceMax over ONLY the innermost axis: kept as a reduction (NOT pooled)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_max_innermost_axis_not_pooled
// A single-axis ReduceMax on the innermost (feature/class) dim must stay an
// ONNXReduceMax so it lowers to xir.qlinear_reduction_max, matching the xmodel
// flow. Lowering it to a spatial MaxPool would fold away the non-reduced 1600
// dim. Mirrors the yolov8l-worldv2 attention ReduceMax (1x8x1600x80, axis 3).
func.func @reduce_max_innermost_axis_not_pooled(%arg0: tensor<1x8x1600x80x!quant.uniform<i8:f32, 0.26:0>>) -> tensor<1x8x1600x1x!quant.uniform<i8:f32, 0.11:0>> {
    %0 = onnx.Constant dense<[3]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x1600x80x!quant.uniform<i8:f32, 0.26:0>>, tensor<1xi64>) -> tensor<1x8x1600x1x!quant.uniform<i8:f32, 0.11:0>>
    return %1 : tensor<1x8x1600x1x!quant.uniform<i8:f32, 0.11:0>>
}
// CHECK: "onnx.ReduceMax"
// CHECK-NOT: onnx.MaxPoolSingleOut

// -----

// CHECK-LABEL: @reduce_max_innermost_axis_negative_not_pooled
// Same as above but with a negative axis (-1); normalized to the innermost dim.
func.func @reduce_max_innermost_axis_negative_not_pooled(%arg0: tensor<1x3x4x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x4x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[-1]> : tensor<1xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x4x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x4x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x4x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMax"
// CHECK-NOT: onnx.MaxPoolSingleOut

// -----

//===----------------------------------------------------------------------===//
// Edge Cases: Trivial reduction (dimension = 1) should be simplified
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_trivial_axis
// NCHW: tensor<1x3x1x4> - reduce axis [2] where H=1 (trivial)
func.func @reduce_mean_trivial_axis(%arg0: tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x3x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// Trivial reduction (dim=1) should be replaced with input directly
// CHECK-NOT: onnx.ReduceMean
// CHECK-NOT: onnx.AveragePool
// CHECK: return %arg0

// -----

//===----------------------------------------------------------------------===//
// Different tensor shapes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_larger_spatial
// NCHW: tensor<1x64x16x16> - reduce axes [2, 3] (H, W)
// Full-spatial reduction -> emitted as ONNXGlobalAveragePool with output
// quant scale preserved (PR #780 invariant).
func.func @reduce_mean_larger_spatial(%arg0: tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.1:1>>
    return %1 : tensor<1x64x1x1x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.GlobalAveragePool"
// CHECK-SAME: !quant.uniform<i8:f32, 1.000000e-01:1>
// CHECK-NOT: onnx.ReduceMean
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_max_larger_spatial
// NCHW: tensor<1x64x16x16> - reduce axes [2, 3] (H, W)
func.func @reduce_max_larger_spatial(%arg0: tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x64x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x64x1x1x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK-NOT: onnx.ReduceMax

// -----

// CHECK-LABEL: @reduce_sum_larger_spatial
// NCHW: tensor<1x32x8x8> - reduce axes [2, 3] (H, W)
// Full-spatial ReduceSum is also no longer converted to pool; native
// qlinear_reduction_sum handles it downstream.
func.func @reduce_sum_larger_spatial(%arg0: tensor<1x32x8x8x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x32x1x1x!quant.uniform<i8:f32, 3.2:0>> {
    %0 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x32x8x8x!quant.uniform<i8:f32, 0.05:0>>, tensor<2xi64>) -> tensor<1x32x1x1x!quant.uniform<i8:f32, 3.2:0>>
    return %1 : tensor<1x32x1x1x!quant.uniform<i8:f32, 3.2:0>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: "onnx.AveragePool"
// CHECK-NOT: "onnx.Mul"

// -----

//===----------------------------------------------------------------------===//
// ReduceMeanV13 → AveragePool Tests (attribute-based axes, from
// GlobalAveragePool canonicalization)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @reduce_mean_v13_spatial_hw
// NCHW: tensor<1x3x4x4> - N=1, C=3, H=4, W=4
// Reduce axes [2, 3] (H, W) -> output tensor<1x3x1x1>
// V13 (attribute-based axes) full-spatial reduction -> ONNXGlobalAveragePool.
func.func @reduce_mean_v13_spatial_hw(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x1x1x!quant.uniform<i8:f32, 0.1:1>>
    return %0 : tensor<1x3x1x1x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.GlobalAveragePool"
// CHECK-SAME: !quant.uniform<i8:f32, 1.000000e-01:1>>
// CHECK-NOT: onnx.ReduceMeanV13
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_mean_v13_psv_global_avgpool
// Regression test for the PSV global-average-pool path.
// NCHW: tensor<1x256x64x64> - reduce axes [2, 3] -> tensor<1x256x1x1>.
// Previously this was lowered to ONNXAveragePool with a flattened
// kernel = [1, 4096] (forced by MAX_KERNEL_SIZE = 16), which downstream
// xir lowered to qlinear_pool {kernel=[1,4096], global=true} -- bit-different
// from the golden xmodel flow (kernel=[64,64], global=true) and the cause of
// the accuracy regression observed after PR #780. The fast path now emits
// ONNXGlobalAveragePool directly so the natural [64, 64] kernel is preserved
// all the way through to xir.qlinear_pool, matching legacy ReplaceQDQPoolPass.
func.func @reduce_mean_v13_psv_global_avgpool(%arg0: tensor<1x256x64x64x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x256x1x1x!quant.uniform<u8:f32, 0.1:1>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x256x64x64x!quant.uniform<u8:f32, 0.05:0>>) -> tensor<1x256x1x1x!quant.uniform<u8:f32, 0.1:1>>
    return %0 : tensor<1x256x1x1x!quant.uniform<u8:f32, 0.1:1>>
}
// CHECK: "onnx.GlobalAveragePool"
// CHECK-SAME: tensor<1x256x64x64
// CHECK-SAME: -> tensor<1x256x1x1
// CHECK-SAME: !quant.uniform<u8:f32, 1.000000e-01:1>>
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.AveragePool
// CHECK-NOT: onnx.ReduceMeanV13

// -----

// CHECK-LABEL: @reduce_mean_v13_keepdims_false_no_fast_path
// keepdims=false produces a rank-2 output, so the GlobalAveragePool fast path
// is NOT taken. The original AveragePool + Reshape flatten path still applies.
func.func @reduce_mean_v13_keepdims_false_no_fast_path(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.1:1>> {
    %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 0 : si64} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x!quant.uniform<i8:f32, 0.1:1>>
    return %0 : tensor<1x3x!quant.uniform<i8:f32, 0.1:1>>
}
// CHECK: "onnx.AveragePool"
// CHECK-NOT: onnx.GlobalAveragePool
// CHECK-NOT: onnx.ReduceMeanV13

// -----

// Channel-axis (axis=1) ReduceMean / ReduceSum is not spatial -- not converted.

// CHECK-LABEL: @reduce_mean_channel_axis_rank4_no_convert
func.func @reduce_mean_channel_axis_rank4_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_sum_channel_axis_rank4_no_convert
func.func @reduce_sum_channel_axis_rank4_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceSum"
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_mean_channel_axis_rank4_keepdims_false_no_convert
func.func @reduce_mean_channel_axis_rank4_keepdims_false_no_convert(%arg0: tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64} : (tensor<1x8x4x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x4x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool

// -----

// CHECK-LABEL: @reduce_mean_rank3_axis1_no_convert
func.func @reduce_mean_rank3_axis1_no_convert(%arg0: tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>> {
    %0 = onnx.Constant dense<[1]> : tensor<1xi64>
    %1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 1 : si64} : (tensor<1x8x4x!quant.uniform<i8:f32, 0.05:0>>, tensor<1xi64>) -> tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>>
    return %1 : tensor<1x1x4x!quant.uniform<i8:f32, 0.05:0>>
}
// CHECK: "onnx.ReduceMean"
// CHECK-NOT: onnx.AveragePool
