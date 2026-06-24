// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @test_normalize_conv_autopad_same_upper_3x3
// CHECK-SAME:  ([[X:%.+]]: tensor<1x3x28x28xf32>, [[W:%.+]]: tensor<8x3x3x3xf32>, [[B:%.+]]: tensor<8xf32>)
func.func @test_normalize_conv_autopad_same_upper_3x3(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x28x28xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x28x28xf32>
  onnx.Return %0 : tensor<1x8x28x28xf32>
  // SAME_UPPER 3x3 stride-1: outputSize=28, sumOfPad=(28-1)*1+3-28=2, pads=[1,1,1,1]
  // CHECK: "onnx.Conv"([[X]], [[W]], [[B]]) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [1, 1, 1, 1]
}

// -----

// CHECK-LABEL: func.func @test_normalize_conv_autopad_same_lower_3x3
func.func @test_normalize_conv_autopad_same_lower_3x3(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x28x28xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_LOWER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x28x28xf32>
  onnx.Return %0 : tensor<1x8x28x28xf32>
  // SAME_LOWER 3x3 stride-1: sumOfPad=2 (even), so split is symmetric -> pads=[1,1,1,1]
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [1, 1, 1, 1]
}

// -----

// CHECK-LABEL: func.func @test_normalize_conv_autopad_valid
func.func @test_normalize_conv_autopad_valid(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x26x26xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "VALID", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x26x26xf32>
  onnx.Return %0 : tensor<1x8x26x26xf32>
  // VALID: all zeros
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [0, 0, 0, 0]
}

// -----

// CHECK-LABEL: func.func @test_normalize_conv_autopad_same_upper_1x1
func.func @test_normalize_conv_autopad_same_upper_1x1(
    %x: tensor<1x32x720x1280xf32>, %w: tensor<25x32x1x1xf32>, %b: tensor<25xf32>) -> tensor<1x25x720x1280xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], strides = [1, 1]} :
      (tensor<1x32x720x1280xf32>, tensor<25x32x1x1xf32>, tensor<25xf32>) -> tensor<1x25x720x1280xf32>
  onnx.Return %0 : tensor<1x25x720x1280xf32>
  // 1x1 kernel always produces zero padding regardless of auto_pad
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [0, 0, 0, 0]
}

// -----

// CHECK-LABEL: func.func @test_normalize_conv_autopad_same_upper_stride2
func.func @test_normalize_conv_autopad_same_upper_stride2(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x14x14xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [2, 2]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x14x14xf32>
  onnx.Return %0 : tensor<1x8x14x14xf32>
  // SAME_UPPER 3x3 stride-2: outputSize=ceil(28/2)=14,
  // sumOfPad=(14-1)*2+(3-1)*1+1-28=1 (odd); SAME_UPPER adds extra at end
  // -> pads=[0,0,1,1]
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [0, 0, 1, 1]
}

// -----

// Dynamic spatial dims: pattern must NOT fire for SAME_UPPER.
// CHECK-LABEL: func.func @test_normalize_conv_autopad_dynamic_no_rewrite
func.func @test_normalize_conv_autopad_dynamic_no_rewrite(
    %x: tensor<1x3x?x?xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x?x?xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x3x?x?xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x?x?xf32>
  onnx.Return %0 : tensor<1x8x?x?xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "SAME_UPPER"
  // CHECK-NOT: pads
}

// -----

// NOTSET with explicit pads already present: no rewrite.
// CHECK-LABEL: func.func @test_normalize_conv_notset_already_normalised
func.func @test_normalize_conv_notset_already_normalised(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x28x28xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x28x28xf32>
  onnx.Return %0 : tensor<1x8x28x28xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [1, 1, 1, 1]
}

// -----

// NOTSET with no pads attribute: pattern fires and fills explicit zero pads.
// CHECK-LABEL: func.func @test_normalize_conv_notset_no_pads
func.func @test_normalize_conv_notset_no_pads(
    %x: tensor<1x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<1x8x26x26xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x26x26xf32>
  onnx.Return %0 : tensor<1x8x26x26xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [0, 0, 0, 0]
}

// -----

// Unranked input (tensor<*xf32>): hasShapeAndRank(X) fails, pattern must not
// fire even though auto_pad is SAME_UPPER.
// CHECK-LABEL: func.func @test_normalize_conv_autopad_unranked_input_no_rewrite
func.func @test_normalize_conv_autopad_unranked_input_no_rewrite(
    %x: tensor<*xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<*xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "SAME_UPPER"
  // CHECK-NOT: pads
}

// -----

// Unranked weight (tensor<*xf32>): hasShapeAndRank(W) fails at the first
// guard, before even inspecting X. Pattern must not fire.
// CHECK-LABEL: func.func @test_normalize_conv_autopad_unranked_weight_no_rewrite
func.func @test_normalize_conv_autopad_unranked_weight_no_rewrite(
    %x: tensor<1x3x28x28xf32>, %w: tensor<*xf32>, %b: tensor<8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<1x3x28x28xf32>, tensor<*xf32>, tensor<8xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "SAME_UPPER"
  // CHECK-NOT: pads
}

// -----

// Dynamic non-spatial dims (batch and channel are '?', spatial dims are
// static): the pattern only reads xShape[2+i], so it must still fire.
// CHECK-LABEL: func.func @test_normalize_conv_autopad_dynamic_nonspatial_fires
func.func @test_normalize_conv_autopad_dynamic_nonspatial_fires(
    %x: tensor<?x3x28x28xf32>, %w: tensor<8x3x3x3xf32>, %b: tensor<8xf32>) -> tensor<?x8x28x28xf32> {
  %0 = "onnx.Conv"(%x, %w, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [1, 1]} :
      (tensor<?x3x28x28xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<?x8x28x28xf32>
  onnx.Return %0 : tensor<?x8x28x28xf32>
  // Spatial dims are static: pads can be computed.
  // CHECK: "onnx.Conv"({{.*}}) {auto_pad = "NOTSET"
  // CHECK-SAME: pads = [1, 1, 1, 1]
}
