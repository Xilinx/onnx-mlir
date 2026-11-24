// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --implicit-check-not=tosa.reshape
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -canonicalize -cse %s -split-input-file | FileCheck %s --check-prefix=LOWERED --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -canonicalize -cse %s -split-input-file | FileCheck %s --check-prefix=LOWERED --implicit-check-not='"onnx.Reshape"'

// Auxiliary reshapes remain in ONNX when Reshape is excluded and lower
// transitively when its conversion is enabled.

func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @test_gemm_to_matmul(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x5xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x5x4xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x4xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<3x4xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_gemm_to_matmul(
// LOWERED: tosa.matmul
// LOWERED: return

// -----

func.func @test_onnx_to_matmul2d(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: func.func @test_onnx_to_matmul2d(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x4x8xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x8x16xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<4x16xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_onnx_to_matmul2d(
// LOWERED: tosa.matmul
// LOWERED: return

// -----

func.func @test_onnx_1d(%arg0 : tensor<6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: func.func @test_onnx_1d(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x6xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<6x1xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x1x6xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x6x1xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<f32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_onnx_1d(
// LOWERED: tosa.matmul
// LOWERED: return

// -----

func.func @test_onnx_to_matmul4d_non_broadcastable(%arg0 : tensor<4x1x5x6xf32>, %arg1 : tensor<1x3x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x1x5x6xf32>, tensor<1x3x6x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: func.func @test_onnx_to_matmul4d_non_broadcastable(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x20x6xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x6x21xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<4x5x3x7xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_onnx_to_matmul4d_non_broadcastable(
// LOWERED: tosa.matmul
// LOWERED: return

// -----

func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple(%arg0: tensor<32x24x112x112xf32>, %arg1 : tensor<48x1x3x3xf32>, %arg2: tensor<48xf32>) ->  tensor<32x48x112x112xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 24 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1395", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<32x24x112x112xf32>, tensor<48x1x3x3xf32>, tensor<48xf32>) -> tensor<32x48x112x112xf32>
  return %0 : tensor<32x48x112x112xf32>
}

// CHECK-LABEL: func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<3x3x24x2xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple(
// LOWERED: tosa.depthwise_conv2d
// LOWERED: return

// -----

func.func @test_batchnorm_f32(%arg0: tensor<100x3x10x10xf32>) -> tensor<100x3x10x10xf32> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x10x10xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<100x3x10x10xf32>
    return %4 : tensor<100x3x10x10xf32>
}

// CHECK-LABEL: func.func @test_batchnorm_f32(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x1x1xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x1x1xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x1x1xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x1x1xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_batchnorm_f32(
// LOWERED: tosa.rsqrt
// LOWERED: return

// -----

func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()

}

// CHECK-LABEL: func.func @test_gather_axis1(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x3x3xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x2xi32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x2x3xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_gather_axis1(
// LOWERED: tosa.gather
// LOWERED: return

// -----

func.func @test_gather_like_slice(%arg0 : tensor<3x3xf32>) -> tensor<3xf32> {
  %indices = onnx.Constant dense<0> : tensor<i64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<i64>) -> tensor<3xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()

}

// CHECK-LABEL: func.func @test_gather_like_slice(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<3xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_gather_like_slice(
// LOWERED: tosa.slice
// LOWERED: return

// -----

func.func @slice_just_steps(%arg0: tensor<100x200xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32>
}

// CHECK-LABEL: func.func @slice_just_steps(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<20x5x20x10xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<20x20xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @slice_just_steps(
// LOWERED: tosa.slice
// LOWERED: return

// -----

func.func @test_expand_new_dims_out(%arg0: tensor<1x64x1xf32>) -> tensor<64x64x64x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[64, 64, 64, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<1x64x1xf32>, tensor<4xi64>) -> tensor<64x64x64x64xf32>
  return %1 : tensor<64x64x64x64xf32>
}

// CHECK-LABEL: func.func @test_expand_new_dims_out(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x64x1x1xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_expand_new_dims_out(
// LOWERED: tosa.tile
// LOWERED: return

// -----

func.func @test_pad_f32_non_constant_1Dpadval(%arg0: tensor<20x16x44x32xf32>, %arg1: tensor<1xf32>) ->  tensor<24x22x52x42xf32>     {
    %noval = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>} : () -> tensor<8xi64>
    %2 = "onnx.Pad"(%arg0, %0, %arg1, %noval) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<24x22x52x42xf32>
    return %2 :   tensor<24x22x52x42xf32>
}

// CHECK-LABEL: func.func @test_pad_f32_non_constant_1Dpadval(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<f32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_pad_f32_non_constant_1Dpadval(
// LOWERED: tosa.pad
// LOWERED: return

// -----

func.func @test_quantizeLinear_per_axis(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xf32>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}

// CHECK-LABEL: func.func @test_quantizeLinear_per_axis(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x2xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x2xi8>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @test_quantizeLinear_per_axis(
// LOWERED: tosa.mul
// LOWERED: return

// -----

func.func @quantize_scalar(%arg0 : tensor<f32>) -> tensor<i8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<f32>, tensor<f32>, tensor<i8>) -> tensor<i8>
  return %2 : tensor<i8>
}

// CHECK-LABEL: func.func @quantize_scalar(
// CHECK: "onnx.Reshape"{{.*}} tensor<0xi64>) -> tensor<f32>
// CHECK: "onnx.Reshape"{{.*}} tensor<0xi64>) -> tensor<i8>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @quantize_scalar(
// LOWERED: tosa.mul
// LOWERED: return

// -----

func.func @per_axis(%arg0: tensor<8x2xi8>) -> tensor<8x2xf32> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xi8>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xf32>
  return %2 : tensor<8x2xf32>
}

// CHECK-LABEL: func.func @per_axis(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x2xi8>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<1x2xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @per_axis(
// LOWERED: tosa.mul
// LOWERED: return

// -----

func.func @dequantize_scalar(%arg0 : tensor<i8>) -> tensor<f32> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: func.func @dequantize_scalar(
// CHECK: "onnx.Reshape"{{.*}} tensor<0xi64>) -> tensor<i8>
// CHECK: "onnx.Reshape"{{.*}} tensor<0xi64>) -> tensor<f32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @dequantize_scalar(
// LOWERED: tosa.mul
// LOWERED: return

// -----

func.func @reduce_min_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMin"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @reduce_min_keepdims_false(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<2x5xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @reduce_min_keepdims_false(
// LOWERED: tosa.reduce_min
// LOWERED: return

// -----

func.func @reduce_max_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @reduce_max_keepdims_false(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<2x5xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @reduce_max_keepdims_false(
// LOWERED: tosa.reduce_max
// LOWERED: return

// -----

func.func @reduce_prod_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceProd"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @reduce_prod_keepdims_false(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<2x5xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @reduce_prod_keepdims_false(
// LOWERED: tosa.reduce_prod
// LOWERED: return

// -----

func.func @reduce_sum_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @reduce_sum_keepdims_false(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<2x5xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @reduce_sum_keepdims_false(
// LOWERED: tosa.reduce_sum
// LOWERED: return

// -----

func.func @reduce_mean_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func.func @reduce_mean_keepdims_false(
// CHECK: "onnx.Reshape"{{.*}} -> tensor<2x5xf32>
// CHECK-NOT: "onnx.Reshape"
// CHECK: return
// LOWERED-LABEL: func.func @reduce_mean_keepdims_false(
// LOWERED: tosa.reduce_sum
// LOWERED: return
