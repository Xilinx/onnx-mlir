// RUN: onnx-mlir-opt --transform-5d-transpose-to-4d --shape-inference %s -split-input-file | FileCheck %s

// A six-dimensional spatial merge can have multiple runtime extents.
// Merging identity dimensions must not multiply dynamic sentinels or put them
// in a constant Reshape operand.
// CHECK-LABEL: func.func @dynamic_identity_dimensions
// CHECK-NOT: onnx.Reshape
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 1, 3, 2, 4, 5]} : (tensor<?x?x2x?x2x1152xf32>) -> tensor<?x?x?x2x2x1152xf32>
// CHECK-NOT: onnx.Reshape
// CHECK: return %[[TRANSPOSE]]
func.func @dynamic_identity_dimensions(%input: tensor<?x?x2x?x2x1152xf32>) -> tensor<?x?x?x2x2x1152xf32> {
  %transpose = "onnx.Transpose"(%input) {perm = [0, 1, 3, 2, 4, 5]} : (tensor<?x?x2x?x2x1152xf32>) -> tensor<?x?x?x2x2x1152xf32>
  return %transpose : tensor<?x?x?x2x2x1152xf32>
}

// -----

// The second strategy can merge the static dimensions 0 and 1, but the other
// runtime dimension still cannot be copied into either constant reshape.
// CHECK-LABEL: func.func @dynamic_dimension_outside_merged_pair
// CHECK-NOT: onnx.Reshape
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%arg0) {perm = [2, 0, 1, 4, 3]} : (tensor<2x3x?x5x7xf32>) -> tensor<?x2x3x7x5xf32>
// CHECK-NOT: onnx.Reshape
// CHECK: return %[[TRANSPOSE]]
func.func @dynamic_dimension_outside_merged_pair(%input: tensor<2x3x?x5x7xf32>) -> tensor<?x2x3x7x5xf32> {
  %transpose = "onnx.Transpose"(%input) {perm = [2, 0, 1, 4, 3]} : (tensor<2x3x?x5x7xf32>) -> tensor<?x2x3x7x5xf32>
  return %transpose : tensor<?x2x3x7x5xf32>
}

// -----

// A fully static transpose remains eligible for the consecutive-pair rewrite.
// CHECK-LABEL: func.func @static_consecutive_pair
// CHECK: %[[INPUT_RESHAPE:.*]] = "onnx.Reshape"(%arg0, %{{.*}}) {{.*}} : (tensor<2x3x4x5x7xf32>, tensor<4xi64>) -> tensor<6x4x5x7xf32>
// CHECK: %[[TRANSPOSE:.*]] = "onnx.Transpose"(%[[INPUT_RESHAPE]]) {perm = [1, 0, 3, 2]} : (tensor<6x4x5x7xf32>) -> tensor<4x6x7x5xf32>
// CHECK: %[[OUTPUT_RESHAPE:.*]] = "onnx.Reshape"(%[[TRANSPOSE]], %{{.*}}) {{.*}} : (tensor<4x6x7x5xf32>, tensor<5xi64>) -> tensor<4x2x3x7x5xf32>
// CHECK: return %[[OUTPUT_RESHAPE]]
func.func @static_consecutive_pair(%input: tensor<2x3x4x5x7xf32>) -> tensor<4x2x3x7x5xf32> {
  %transpose = "onnx.Transpose"(%input) {perm = [2, 0, 1, 4, 3]} : (tensor<2x3x4x5x7xf32>) -> tensor<4x2x3x7x5xf32>
  return %transpose : tensor<4x2x3x7x5xf32>
}
