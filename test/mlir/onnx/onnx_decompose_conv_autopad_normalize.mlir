// RUN: onnx-mlir-opt --decompose-onnx=enable-conv-autopad-normalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// VALID auto_pad with no pads -> explicit zero pads + auto_pad NOTSET.
func.func @test_conv_autopad_valid(%arg0: tensor<1x3x512x512xf32>, %arg1: tensor<1024x3x16x16xf32>, %arg2: tensor<1024xf32>) -> tensor<1x1024x32x32xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "VALID", group = 1 : si64, kernel_shape = [16, 16], strides = [16, 16]} : (tensor<1x3x512x512xf32>, tensor<1024x3x16x16xf32>, tensor<1024xf32>) -> tensor<1x1024x32x32xf32>
  return %0 : tensor<1x1024x32x32xf32>

// CHECK-LABEL:  func.func @test_conv_autopad_valid
// CHECK:           "onnx.Conv"
// CHECK-SAME:      auto_pad = "NOTSET"
// CHECK-SAME:      pads = [0, 0, 0, 0]

// DISABLED-LABEL:  func.func @test_conv_autopad_valid
// DISABLED:           "onnx.Conv"
// DISABLED-SAME:      auto_pad = "VALID"
// DISABLED-NOT:       pads
}

// -----

// SAME_UPPER auto_pad: stride 1, kernel 3 -> symmetric pad [1,1,1,1].
func.func @test_conv_autopad_same_upper(%arg0: tensor<1x3x8x8xf32>, %arg1: tensor<4x3x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_UPPER", group = 1 : si64, kernel_shape = [3, 3], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>

// CHECK-LABEL:  func.func @test_conv_autopad_same_upper
// CHECK:           "onnx.Conv"
// CHECK-SAME:      auto_pad = "NOTSET"
// CHECK-SAME:      pads = [1, 1, 1, 1]
}

// -----

// SAME_LOWER auto_pad: stride 2, kernel 3, input 8 -> odd total pad, extra at
// the beginning. outputSize=ceil(8/2)=4, sumOfPad=(4-1)*2+3-8=1 -> begin=1,end=0.
func.func @test_conv_autopad_same_lower(%arg0: tensor<1x3x8x8xf32>, %arg1: tensor<4x3x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x4x4xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_LOWER", group = 1 : si64, kernel_shape = [3, 3], strides = [2, 2]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>

// CHECK-LABEL:  func.func @test_conv_autopad_same_lower
// CHECK:           "onnx.Conv"
// CHECK-SAME:      auto_pad = "NOTSET"
// CHECK-SAME:      pads = [1, 1, 0, 0]
}

// -----

// NOTSET with explicit pads is left untouched even when the flag is on.
func.func @test_conv_notset_untouched(%arg0: tensor<1x3x8x8xf32>, %arg1: tensor<4x3x3x3xf32>, %arg2: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>

// CHECK-LABEL:  func.func @test_conv_notset_untouched
// CHECK:           "onnx.Conv"
// CHECK-SAME:      auto_pad = "NOTSET"
// CHECK-SAME:      pads = [1, 1, 1, 1]

// DISABLED-LABEL:  func.func @test_conv_notset_untouched
// DISABLED:           "onnx.Conv"
// DISABLED-SAME:      auto_pad = "NOTSET"
// DISABLED-SAME:      pads = [1, 1, 1, 1]
}
