// RUN: onnx-mlir-opt --split-input-file --transfer-resize-linear-to-dwconv %s | FileCheck %s --implicit-check-not=onnx.Conv
// RUN: sed 's/bf16/f32/g' %s | onnx-mlir-opt --split-input-file --transfer-resize-linear-to-dwconv | FileCheck %s --implicit-check-not=onnx.Conv
// RUN: sed 's/bf16/f16/g' %s | onnx-mlir-opt --split-input-file --transfer-resize-linear-to-dwconv | FileCheck %s --implicit-check-not=onnx.Conv

// CHECK-LABEL: func.func @resize_half_pixel_scale8
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK-SAME: coordinate_transformation_mode = "half_pixel"
// CHECK: return %[[RESULT]]
func.func @resize_half_pixel_scale8(%input: tensor<1x64x16x16xbf16>) -> tensor<1x64x128x128xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 8.0, 8.0]> : tensor<4xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear", coordinate_transformation_mode = "half_pixel"} : (tensor<1x64x16x16xbf16>, none, tensor<4xf32>, none) -> tensor<1x64x128x128xbf16>
  return %result : tensor<1x64x128x128xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_asymmetric_scale12
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK-SAME: coordinate_transformation_mode = "asymmetric"
// CHECK: return %[[RESULT]]
func.func @resize_asymmetric_scale12(%input: tensor<1x3x32x32xbf16>) -> tensor<1x3x384x384xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 12.0, 12.0]> : tensor<4xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear", coordinate_transformation_mode = "asymmetric"} : (tensor<1x3x32x32xbf16>, none, tensor<4xf32>, none) -> tensor<1x3x384x384xbf16>
  return %result : tensor<1x3x384x384xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_asymmetric_scale24_sizes
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK-SAME: coordinate_transformation_mode = "asymmetric"
// CHECK: return %[[RESULT]]
func.func @resize_asymmetric_scale24_sizes(%input: tensor<1x3x16x16xbf16>) -> tensor<1x3x384x384xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = onnx.Constant dense<[1, 3, 384, 384]> : tensor<4xi64>
  %result = "onnx.Resize"(%input, %none, %none, %sizes) {mode = "linear", coordinate_transformation_mode = "asymmetric"} : (tensor<1x3x16x16xbf16>, none, none, tensor<4xi64>) -> tensor<1x3x384x384xbf16>
  return %result : tensor<1x3x384x384xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_pytorch_half_pixel_scale2
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK-SAME: coordinate_transformation_mode = "pytorch_half_pixel"
// CHECK: return %[[RESULT]]
func.func @resize_pytorch_half_pixel_scale2(%input: tensor<1x3x6x6xbf16>) -> tensor<1x3x12x12xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear", coordinate_transformation_mode = "pytorch_half_pixel"} : (tensor<1x3x6x6xbf16>, none, tensor<4xf32>, none) -> tensor<1x3x12x12xbf16>
  return %result : tensor<1x3x12x12xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_linear_downsample
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK: return %[[RESULT]]
func.func @resize_linear_downsample(%input: tensor<1x3x12x12xbf16>) -> tensor<1x3x6x6xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 0.5, 0.5]> : tensor<4xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear", coordinate_transformation_mode = "half_pixel"} : (tensor<1x3x12x12xbf16>, none, tensor<4xf32>, none) -> tensor<1x3x6x6xbf16>
  return %result : tensor<1x3x6x6xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_trilinear_upsample
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK-SAME: mode = "trilinear"
// CHECK: return %[[RESULT]]
func.func @resize_trilinear_upsample(%input: tensor<1x3x4x4x4xbf16>) -> tensor<1x3x8x8x8xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0, 2.0]> : tensor<5xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "trilinear", coordinate_transformation_mode = "half_pixel"} : (tensor<1x3x4x4x4xbf16>, none, tensor<5xf32>, none) -> tensor<1x3x8x8x8xbf16>
  return %result : tensor<1x3x8x8x8xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_trilinear_downsample_sizes
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK-SAME: mode = "trilinear"
// CHECK: return %[[RESULT]]
func.func @resize_trilinear_downsample_sizes(%input: tensor<1x3x8x8x8xbf16>) -> tensor<1x3x4x4x4xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = onnx.Constant dense<[1, 3, 4, 4, 4]> : tensor<5xi64>
  %result = "onnx.Resize"(%input, %none, %none, %sizes) {mode = "trilinear", coordinate_transformation_mode = "asymmetric"} : (tensor<1x3x8x8x8xbf16>, none, none, tensor<5xi64>) -> tensor<1x3x4x4x4xbf16>
  return %result : tensor<1x3x4x4x4xbf16>
}

// -----

// CHECK-LABEL: func.func @resize_quantized_upsample
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK: return %[[RESULT]]
func.func @resize_quantized_upsample(%input: tensor<1x3x6x6x!quant.uniform<u8:f32, 0.125:128>>) -> tensor<1x3x12x12x!quant.uniform<u8:f32, 0.125:128>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear", coordinate_transformation_mode = "half_pixel"} : (tensor<1x3x6x6x!quant.uniform<u8:f32, 0.125:128>>, none, tensor<4xf32>, none) -> tensor<1x3x12x12x!quant.uniform<u8:f32, 0.125:128>>
  return %result : tensor<1x3x12x12x!quant.uniform<u8:f32, 0.125:128>>
}

// -----

// CHECK-LABEL: func.func @resize_quantized_downsample
// CHECK: %[[RESULT:.*]] = "onnx.Resize"
// CHECK: return %[[RESULT]]
func.func @resize_quantized_downsample(%input: tensor<1x3x12x12x!quant.uniform<u8:f32, 0.125:128>>) -> tensor<1x3x6x6x!quant.uniform<u8:f32, 0.125:128>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = onnx.Constant dense<[1.0, 1.0, 0.5, 0.5]> : tensor<4xf32>
  %result = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear", coordinate_transformation_mode = "half_pixel"} : (tensor<1x3x12x12x!quant.uniform<u8:f32, 0.125:128>>, none, tensor<4xf32>, none) -> tensor<1x3x6x6x!quant.uniform<u8:f32, 0.125:128>>
  return %result : tensor<1x3x6x6x!quant.uniform<u8:f32, 0.125:128>>
}
