// RUN: onnx-mlir-opt --convert-to-channel-last %s -split-input-file | FileCheck %s

// A failed match must not create a transpose or partially permuted constants.
// Otherwise greedy rewriting keeps creating and deleting these unused ops and
// fails to converge.
// Keep runtime sizes unchanged: fixed input dimensions do not make the values
// of the sizes tensor constant.
// CHECK-LABEL: func.func @runtime_sizes
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: %[[RESIZE:.*]] = "onnx.Resize"(%arg0, %{{.*}}, %{{.*}}, %arg1)
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: onnx.Return %[[RESIZE]]
func.func @runtime_sizes(%input: tensor<1x1152x27x27xf32>, %sizes: tensor<4xi64>) -> tensor<1x1152x?x?xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %resize = "onnx.Resize"(%input, %none, %none, %sizes) {mode = "linear"} : (tensor<1x1152x27x27xf32>, none, none, tensor<4xi64>) -> tensor<1x1152x?x?xf32>
  onnx.Return %resize : tensor<1x1152x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @runtime_scales
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: %[[RESIZE:.*]] = "onnx.Resize"(%arg0, %{{.*}}, %arg1, %{{.*}})
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: onnx.Return %[[RESIZE]]
func.func @runtime_scales(%input: tensor<1x3x4x4xf32>, %scales: tensor<4xf32>) -> tensor<1x3x?x?xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %resize = "onnx.Resize"(%input, %none, %scales, %none) {mode = "linear"} : (tensor<1x3x4x4xf32>, none, tensor<4xf32>, none) -> tensor<1x3x?x?xf32>
  onnx.Return %resize : tensor<1x3x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @runtime_roi
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: %[[RESIZE:.*]] = "onnx.Resize"(%arg0, %arg1, %{{.*}}, %{{.*}})
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: onnx.Return %[[RESIZE]]
func.func @runtime_roi(%input: tensor<1x3x4x4xf32>, %roi: tensor<8xf32>) -> tensor<1x3x8x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = onnx.Constant dense<[1, 3, 8, 8]> : tensor<4xi64>
  %resize = "onnx.Resize"(%input, %roi, %none, %sizes) {coordinate_transformation_mode = "tf_crop_and_resize", mode = "linear"} : (tensor<1x3x4x4xf32>, tensor<8xf32>, none, tensor<4xi64>) -> tensor<1x3x8x8xf32>
  onnx.Return %resize : tensor<1x3x8x8xf32>
}

// -----

// In this case the constant ROI would be permuted before sizes were rejected.
// CHECK-LABEL: func.func @constant_roi_runtime_sizes
// CHECK: %[[ROI:.*]] = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 2.000000e-01, 3.000000e-01, 1.000000e+00, 1.000000e+00, 8.000000e-01, 0.699999988]> : tensor<8xf32>
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: %[[RESIZE:.*]] = "onnx.Resize"(%arg0, %[[ROI]], %{{.*}}, %arg1)
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.XFEResize
// CHECK: onnx.Return %[[RESIZE]]
func.func @constant_roi_runtime_sizes(%input: tensor<1x3x4x4xf32>, %sizes: tensor<4xi64>) -> tensor<1x3x?x?xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %roi = onnx.Constant dense<[0.0, 0.0, 0.2, 0.3, 1.0, 1.0, 0.8, 0.7]> : tensor<8xf32>
  %resize = "onnx.Resize"(%input, %roi, %none, %sizes) {coordinate_transformation_mode = "tf_crop_and_resize", mode = "linear"} : (tensor<1x3x4x4xf32>, tensor<8xf32>, none, tensor<4xi64>) -> tensor<1x3x?x?xf32>
  onnx.Return %resize : tensor<1x3x?x?xf32>
}
