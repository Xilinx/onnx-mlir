// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_materialize_conv_defaults_2d
func.func @test_materialize_conv_defaults_2d(
    %x: tensor<1x3x224x224xf32>, %w: tensor<512x3x3x3xf32>, %b: tensor<512xf32>)
    -> tensor<1x512x224x224xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "NOTSET", group = 1 : si64,
      kernel_shape = [3, 3], pads = [1, 1, 1, 1]} :
      (tensor<1x3x224x224xf32>, tensor<512x3x3x3xf32>, tensor<512xf32>)
      -> tensor<1x512x224x224xf32>
  onnx.Return %0 : tensor<1x512x224x224xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: dilations = [1, 1]
  // CHECK-SAME: kernel_shape = [3, 3]
  // CHECK-SAME: pads = [1, 1, 1, 1]
  // CHECK-SAME: strides = [1, 1]
}

// -----

// CHECK-LABEL: func.func @test_materialize_conv_defaults_3d
func.func @test_materialize_conv_defaults_3d(
    %x: tensor<1x2x4x5x7xf32>, %w: tensor<4x2x2x3x5xf32>, %b: tensor<4xf32>)
    -> tensor<1x4x3x3x3xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "NOTSET", group = 1 : si64,
      kernel_shape = [2, 3, 5], pads = [0, 0, 0, 0, 0, 0]} :
      (tensor<1x2x4x5x7xf32>, tensor<4x2x2x3x5xf32>, tensor<4xf32>)
      -> tensor<1x4x3x3x3xf32>
  onnx.Return %0 : tensor<1x4x3x3x3xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: dilations = [1, 1, 1]
  // CHECK-SAME: kernel_shape = [2, 3, 5]
  // CHECK-SAME: pads = [0, 0, 0, 0, 0, 0]
  // CHECK-SAME: strides = [1, 1, 1]
}

// -----

// CHECK-LABEL: func.func @test_materialize_conv_defaults_notset_no_pads
func.func @test_materialize_conv_defaults_notset_no_pads(
    %x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>, %b: tensor<4xf32>)
    -> tensor<1x4x6x6xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "NOTSET", group = 1 : si64,
      kernel_shape = [3, 3]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>)
      -> tensor<1x4x6x6xf32>
  onnx.Return %0 : tensor<1x4x6x6xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: dilations = [1, 1]
  // CHECK-SAME: pads = [0, 0, 0, 0]
  // CHECK-SAME: strides = [1, 1]
}

// -----

// CHECK-LABEL: func.func @test_materialize_conv_defaults_valid
func.func @test_materialize_conv_defaults_valid(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>)
    -> tensor<1x8x26x26xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "VALID", group = 1 : si64,
      kernel_shape = [3, 3]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>)
      -> tensor<1x8x26x26xf32>
  onnx.Return %0 : tensor<1x8x26x26xf32>
  // MaterializeDefaultConvParamsPattern fills defaults; NormalizeConvAutoPadPattern
  // then rewrites VALID to NOTSET with the same zero pads.
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: dilations = [1, 1]
  // CHECK-SAME: pads = [0, 0, 0, 0]
  // CHECK-SAME: strides = [1, 1]
}
