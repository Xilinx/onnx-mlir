// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

func.func @fuse_gather_elements_tile(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<2x3x4xf32> {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @fuse_gather_elements_tile(
// CHECK-NOT: "onnx.Tile"
// CHECK: [[RESHAPED:%.+]] = "onnx.Reshape"(%arg1, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-NEXT: [[GATHER:%.+]] = "onnx.Gather"(%arg0, [[RESHAPED]]) {axis = 1 : si64} : (tensor<2x5x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NOT: "onnx.GatherElements"
// CHECK-NEXT: onnx.Return [[GATHER]] : tensor<2x3x4xf32>

// -----

func.func @fuse_gather_elements_tile_negative_axis(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<2x3x4xf32> {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = -2 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @fuse_gather_elements_tile_negative_axis(
// CHECK-NOT: "onnx.Tile"
// CHECK: [[RESHAPED:%.+]] = "onnx.Reshape"(%arg1, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-NEXT: [[GATHER:%.+]] = "onnx.Gather"(%arg0, [[RESHAPED]]) {axis = -2 : si64} : (tensor<2x5x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NOT: "onnx.GatherElements"
// CHECK-NEXT: onnx.Return [[GATHER]] : tensor<2x3x4xf32>

// -----

// Repeating the GatherElements axis cannot be represented by a 1-D Gather.
func.func @do_not_fuse_repeated_axis(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<2x6x4xf32> {
  %repeats = onnx.Constant dense<[2, 2, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x6x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x6x4xi64>) -> tensor<2x6x4xf32>
  onnx.Return %result : tensor<2x6x4xf32>
}
// CHECK-LABEL: func.func @do_not_fuse_repeated_axis(
// CHECK: [[TILED:%.+]] = "onnx.Tile"
// CHECK: [[RESULT:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]]) {axis = 1 : si64}
// CHECK: onnx.Return [[RESULT]]

// -----

// Gather would retain the full first data dimension, unlike GatherElements.
func.func @do_not_fuse_partial_non_axis_dimension(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<1x3x4xf32> {
  %repeats = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<1x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<1x3x4xi64>) -> tensor<1x3x4xf32>
  onnx.Return %result : tensor<1x3x4xf32>
}
// CHECK-LABEL: func.func @do_not_fuse_partial_non_axis_dimension(
// CHECK: [[TILED:%.+]] = "onnx.Tile"
// CHECK: [[RESULT:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]]) {axis = 1 : si64}
// CHECK: onnx.Return [[RESULT]]
