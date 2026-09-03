// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="grouped-conv-threshold=4" -cse %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="grouped-conv-threshold=4 excluded-ops=Concat" -cse %s -split-input-file | FileCheck %s --check-prefix=EXCLUDE

func.func @test_group_conv_preserves_excluded_concat(%arg0: tensor<5x64x256x256xf32>, %arg1 : tensor<12x16x45x45xf32>, %arg2: tensor<12xf32>) -> tensor<5x12x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
      group = 4 : si64,
      pads = [1, 1, 1, 1],
      strides = [13, 13]
    } : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) -> tensor<5x12x17x17xf32>
  return %0 : tensor<5x12x17x17xf32>

// CHECK-LABEL:   func.func @test_group_conv_preserves_excluded_concat(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<5x64x256x256xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<12x16x45x45xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<12xf32>) -> tensor<5x12x17x17xf32> {
// CHECK:           %[[CONV0:.*]] = tosa.conv2d
// CHECK:           %[[CONV1:.*]] = tosa.conv2d
// CHECK:           %[[CONV2:.*]] = tosa.conv2d
// CHECK:           %[[CONV3:.*]] = tosa.conv2d
// CHECK-NOT:       "onnx.Concat"
// CHECK:           %[[CONCAT:.*]] = tosa.concat %[[CONV0]], %[[CONV1]], %[[CONV2]], %[[CONV3]] {axis = 3 : i32}
// CHECK:           tosa.transpose %[[CONCAT]]
// CHECK:           return

// EXCLUDE-LABEL:   func.func @test_group_conv_preserves_excluded_concat(
// EXCLUDE-SAME:      %[[ARG0:.*]]: tensor<5x64x256x256xf32>,
// EXCLUDE-SAME:      %[[ARG1:.*]]: tensor<12x16x45x45xf32>,
// EXCLUDE-SAME:      %[[ARG2:.*]]: tensor<12xf32>) -> tensor<5x12x17x17xf32> {
// EXCLUDE:           %[[CONV0:.*]] = tosa.conv2d
// EXCLUDE:           %[[CONV1:.*]] = tosa.conv2d
// EXCLUDE:           %[[CONV2:.*]] = tosa.conv2d
// EXCLUDE:           %[[CONV3:.*]] = tosa.conv2d
// EXCLUDE-NOT:       tosa.concat
// EXCLUDE:           %[[CONCAT:.*]] = "onnx.Concat"(%[[CONV0]], %[[CONV1]], %[[CONV2]], %[[CONV3]]) {axis = 3 : si64}
// EXCLUDE:           tosa.transpose %[[CONCAT]]
}
