// RUN: onnx-mlir-opt --onnx-hybrid-transform %s -split-input-file | FileCheck %s

// -----

func.func @test_fuse_conv1x1_basic(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+01> : tensor<8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_without_explicit_dilations(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_without_explicit_dilations
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+01> : tensor<8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_1d(
    %x: tensor<1x3x8xf32>) -> tensor<1x8x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1], group = 1 : si64,
      kernel_shape = [1], pads = [0, 0], strides = [1]} :
      (tensor<1x3x8xf32>, tensor<4x3x1xf32>, tensor<4xf32>) -> tensor<1x4x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1], group = 1 : si64,
      kernel_shape = [3], pads = [0, 0], strides = [1]} :
      (tensor<1x4x8xf32>, tensor<8x4x3xf32>, tensor<8xf32>) -> tensor<1x8x6xf32>
  onnx.Return %c2 : tensor<1x8x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8xf32>) -> tensor<1x8x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<8.000000e+00> : tensor<8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [3], pads = [0, 0], strides = [1]} : (tensor<1x3x8xf32>, tensor<8x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_3d(
    %x: tensor<1x3x8x8x8xf32>) -> tensor<1x8x6x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64,
      kernel_shape = [1, 1, 1], pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} :
      (tensor<1x3x8x8x8xf32>, tensor<4x3x1x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64,
      kernel_shape = [3, 3, 3], pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} :
      (tensor<1x4x8x8x8xf32>, tensor<8x4x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_3d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8x8xf32>) -> tensor<1x8x6x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.600000e+01> : tensor<8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1, 1], group = 1 : si64, kernel_shape = [3, 3, 3], pads = [0, 0, 0, 0, 0, 0], strides = [1, 1, 1]} : (tensor<1x3x8x8x8xf32>, tensor<8x3x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_no_bias(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_no_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, none) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_same_upper_b1_zero(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.0> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
  onnx.Return %c2 : tensor<1x8x8x8xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_same_upper_b1_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x8x8xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_padded_nonzero_b1(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
  onnx.Return %c2 : tensor<1x8x8x8xf32>


// CHECK-LABEL:  func.func @test_no_fuse_padded_nonzero_b1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Conv"([[VAR_4_]], [[VAR_2_]], [[VAR_3_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<1x8x8x8xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_multiple_uses(
    %x: tensor<1x3x8x8xf32>) -> (tensor<1x8x6x6xf32>, tensor<1x4x8x8xf32>) {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
  onnx.Return %c2, %c1 : tensor<1x8x6x6xf32>, tensor<1x4x8x8xf32>


// CHECK-LABEL:  func.func @test_no_fuse_multiple_uses
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> (tensor<1x8x6x6xf32>, tensor<1x4x8x8xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[VAR_1_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_4_]], [[VAR_3_]] : tensor<1x8x6x6xf32>, tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_non_1x1_conv1(
    %x: tensor<1x3x10x10xf32>) -> tensor<1x8x4x4xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x3x3xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x10x10xf32>, tensor<4x3x3x3xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x4x4xf32>
  onnx.Return %c2 : tensor<1x8x4x4xf32>


// CHECK-LABEL:  func.func @test_no_fuse_non_1x1_conv1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x10x10xf32>) -> tensor<1x8x4x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x10x10xf32>, tensor<4x3x3x3xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[VAR_1_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x4x4xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x8x4x4xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_stride2(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x3x3xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x3x3xf32>
  onnx.Return %c2 : tensor<1x8x3x3xf32>


// CHECK-LABEL:  func.func @test_no_fuse_stride2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x3x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[VAR_1_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x3x3xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x8x3x3xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_group(
    %x: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x4x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<4x1x1x1xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<4x4x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 4 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<4x1x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  onnx.Return %c2 : tensor<1x4x8x8xf32>


// CHECK-LABEL:  func.func @test_no_fuse_group
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x4x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x1x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<4x4x1x1xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[VAR_1_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 4 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<4x1x1x1xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_b1_none(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_b1_none
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_b2_none(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_b2_none
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4.000000e+00> : tensor<8x3x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.800000e+01> : tensor<8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_non_constant_w1(
    %x: tensor<1x3x8x8xf32>,
    %w1: tensor<4x3x1x1xf32>) -> tensor<1x8x6x6xf32> {
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_no_fuse_non_constant_w1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x1x1xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[VAR_2_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_non_constant_w2(
    %x: tensor<1x3x8x8xf32>,
    %w2: tensor<8x4x3x3xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_no_fuse_non_constant_w2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<8x4x3x3xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[VAR_2_]], [[PARAM_1_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, none) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_non_constant_b1(
    %x: tensor<1x3x8x8xf32>,
    %b1: tensor<4xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_no_fuse_non_constant_b1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[VAR_1_]], [[VAR_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_non_constant_b2(
    %x: tensor<1x3x8x8xf32>,
    %b2: tensor<8xf32>) -> tensor<1x8x6x6xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
  onnx.Return %c2 : tensor<1x8x6x6xf32>


// CHECK-LABEL:  func.func @test_no_fuse_non_constant_b2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>) -> tensor<1x8x6x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[VAR_2_]], [[PARAM_1_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x6x6xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<1x8x6x6xf32>
// CHECK:         }
}

// -----

func.func @test_fuse_conv1x1_numerical_weight(
    %x: tensor<1x2x6x6xf32>) -> tensor<1x2x4x4xf32> {
  %w1 = onnx.Constant dense<[[[[2.0]], [[3.0]]], [[[4.0]], [[5.0]]]]> : tensor<2x2x1x1xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<2x2x3x3xf32>
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x2x6x6xf32>, tensor<2x2x1x1xf32>, none) -> tensor<1x2x6x6xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x2x6x6xf32>, tensor<2x2x3x3xf32>, none) -> tensor<1x2x4x4xf32>
  onnx.Return %c2 : tensor<1x2x4x4xf32>


// CHECK-LABEL:  func.func @test_fuse_conv1x1_numerical_weight
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x6x6xf32>) -> tensor<1x2x4x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{.}}[{{.}}[6.000000e+00, 6.000000e+00, 6.000000e+00], [6.000000e+00, 6.000000e+00, 6.000000e+00], [6.000000e+00, 6.000000e+00, 6.000000e+00]{{.}}, {{.}}[8.000000e+00, 8.000000e+00, 8.000000e+00], [8.000000e+00, 8.000000e+00, 8.000000e+00], [8.000000e+00, 8.000000e+00, 8.000000e+00]{{.}}{{.}}, {{.}}{{.}}[6.000000e+00, 6.000000e+00, 6.000000e+00], [6.000000e+00, 6.000000e+00, 6.000000e+00], [6.000000e+00, 6.000000e+00, 6.000000e+00]{{.}}, {{.}}[8.000000e+00, 8.000000e+00, 8.000000e+00], [8.000000e+00, 8.000000e+00, 8.000000e+00], [8.000000e+00, 8.000000e+00, 8.000000e+00]{{.}}{{.}}]> : tensor<2x2x3x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x6x6xf32>, tensor<2x2x3x3xf32>, none) -> tensor<1x2x4x4xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x2x4x4xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_dynamic_same_upper_nonzero_b1(
    %x: tensor<1x3x?x?xf32>) -> tensor<1x8x?x?xf32> {
  %w1 = onnx.Constant dense<1.0> : tensor<4x3x1x1xf32>
  %b1 = onnx.Constant dense<0.5> : tensor<4xf32>
  %w2 = onnx.Constant dense<1.0> : tensor<8x4x3x3xf32>
  %b2 = onnx.Constant dense<2.0> : tensor<8xf32>
  %c1 = "onnx.Conv"(%x, %w1, %b1) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], strides = [1, 1]} :
      (tensor<1x3x?x?xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x?x?xf32>
  %c2 = "onnx.Conv"(%c1, %w2, %b2) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x4x?x?xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x?x?xf32>
  onnx.Return %c2 : tensor<1x8x?x?xf32>


// CHECK-LABEL:  func.func @test_no_fuse_dynamic_same_upper_nonzero_b1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x?x?xf32>) -> tensor<1x8x?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<4x3x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<8x4x3x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], strides = [1, 1]} : (tensor<1x3x?x?xf32>, tensor<4x3x1x1xf32>, tensor<4xf32>) -> tensor<1x4x?x?xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Conv"([[VAR_4_]], [[VAR_2_]], [[VAR_3_]]) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], strides = [1, 1]} : (tensor<1x4x?x?xf32>, tensor<8x4x3x3xf32>, tensor<8xf32>) -> tensor<1x8x?x?xf32>
// CHECK:           onnx.Return [[VAR_5_]] : tensor<1x8x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_no_fuse_quantized_conv1x1(
    %x: tensor<1x3x8x8x!quant.uniform<i8:f32, 0.1:0>>,
    %w1: tensor<4x3x1x1x!quant.uniform<i8:f32, 0.1:0>>,
    %w2: tensor<8x4x3x3x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x8x6x6x!quant.uniform<i8:f32, 0.1:0>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %c1 = "onnx.Conv"(%x, %w1, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x3x8x8x!quant.uniform<i8:f32, 0.1:0>>,
       tensor<4x3x1x1x!quant.uniform<i8:f32, 0.1:0>>, none) ->
      tensor<1x4x8x8x!quant.uniform<i8:f32, 0.1:0>>
  %c2 = "onnx.Conv"(%c1, %w2, %none) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} :
      (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.1:0>>,
       tensor<8x4x3x3x!quant.uniform<i8:f32, 0.1:0>>, none) ->
      tensor<1x8x6x6x!quant.uniform<i8:f32, 0.1:0>>
  onnx.Return %c2 : tensor<1x8x6x6x!quant.uniform<i8:f32, 0.1:0>>


// CHECK-LABEL:  func.func @test_no_fuse_quantized_conv1x1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8x!quant.uniform<i8:f32, 1.000000e-01>>, [[PARAM_1_:%.+]]: tensor<4x3x1x1x!quant.uniform<i8:f32, 1.000000e-01>>, [[PARAM_2_:%.+]]: tensor<8x4x3x3x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<1x8x6x6x!quant.uniform<i8:f32, 1.000000e-01>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8x!quant.uniform<i8:f32, 1.000000e-01>>, tensor<4x3x1x1x!quant.uniform<i8:f32, 1.000000e-01>>, none) -> tensor<1x4x8x8x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Conv"([[VAR_1_]], [[PARAM_2_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x4x8x8x!quant.uniform<i8:f32, 1.000000e-01>>, tensor<8x4x3x3x!quant.uniform<i8:f32, 1.000000e-01>>, none) -> tensor<1x8x6x6x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x8x6x6x!quant.uniform<i8:f32, 1.000000e-01>>
// CHECK:         }
}
