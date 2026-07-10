// RUN: onnx-mlir-opt --enable-expand-canonicalization --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file | FileCheck %s --check-prefix=NOOPT

// -----

func.func @expand_same_rank_to_tile(%arg0: tensor<2x1x4xf32>) -> tensor<2x3x4xf32> {
  %shape = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<2x1x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>

// CHECK-LABEL:  func.func @expand_same_rank_to_tile
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x4xf32>) -> tensor<2x3x4xf32> {
// CHECK-NOT:       "onnx.Expand"
// CHECK:           [[REPEATS_:%.+]] = onnx.Constant dense<[1, 3, 1]> : tensor<3xi64>
// CHECK:           [[TILE_:%.+]] = "onnx.Tile"([[PARAM_0_]], [[REPEATS_]]) : (tensor<2x1x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK:           return [[TILE_]] : tensor<2x3x4xf32>
// CHECK:         }

// Disabled by default: the Expand must be preserved.
// NOOPT-LABEL:  func.func @expand_same_rank_to_tile
// NOOPT-NOT:       "onnx.Tile"
// NOOPT:           "onnx.Expand"
}

// -----

func.func @expand_same_rank_dynamic_input_unchanged(%arg0: tensor<2x?x4xf32>) -> tensor<2x3x4xf32> {
  %shape = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<2x?x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>

// CHECK-LABEL:  func.func @expand_same_rank_dynamic_input_unchanged
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?x4xf32>) -> tensor<2x3x4xf32> {
// CHECK-NOT:       "onnx.Tile"
// CHECK:           [[SHAPE_:%.+]] = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
// CHECK:           [[EXPAND_:%.+]] = "onnx.Expand"([[PARAM_0_]], [[SHAPE_]]) : (tensor<2x?x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NOT:       "onnx.Tile"
// CHECK:           return [[EXPAND_]] : tensor<2x3x4xf32>
// CHECK:         }
}

// -----

func.func @expand_rank_increase_to_reshape_tile(%arg0: tensor<3x1xf32>) -> tensor<2x3x4xf32> {
  %shape = onnx.Constant dense<[2, 3, 4]> : tensor<3xi64>
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>

// CHECK-LABEL:  func.func @expand_rank_increase_to_reshape_tile
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x1xf32>) -> tensor<2x3x4xf32> {
// CHECK-DAG:       [[RESHAPE_SHAPE_:%.+]] = onnx.Constant dense<[1, 3, 1]> : tensor<3xi64>
// CHECK-DAG:       [[REPEATS_:%.+]] = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
// CHECK:           [[RESHAPED_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[RESHAPE_SHAPE_]]) {allowzero = 0 : si64} : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<1x3x1xf32>
// CHECK:           [[TILE_:%.+]] = "onnx.Tile"([[RESHAPED_]], [[REPEATS_]]) : (tensor<1x3x1xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NOT:       "onnx.Expand"
// CHECK:           return [[TILE_]] : tensor<2x3x4xf32>
// CHECK:         }

// Disabled by default: the Expand must be preserved (no Reshape/Tile).
// NOOPT-LABEL:  func.func @expand_rank_increase_to_reshape_tile
// NOOPT-NOT:       "onnx.Tile"
// NOOPT-NOT:       "onnx.Reshape"
// NOOPT:           "onnx.Expand"
}

// -----

func.func @expand_rank_increase_dynamic_input_unchanged(%arg0: tensor<?x1xf32>, %shape: tensor<3xi64>) -> tensor<2x?x4xf32> {
  %0 = "onnx.Expand"(%arg0, %shape) : (tensor<?x1xf32>, tensor<3xi64>) -> tensor<2x?x4xf32>
  return %0 : tensor<2x?x4xf32>

// CHECK-LABEL:  func.func @expand_rank_increase_dynamic_input_unchanged
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x1xf32>, [[PARAM_1_:%.+]]: tensor<3xi64>) -> tensor<2x?x4xf32> {
// CHECK-NOT:       "onnx.Reshape"
// CHECK-NOT:       "onnx.Tile"
// CHECK:           [[EXPAND_:%.+]] = "onnx.Expand"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x1xf32>, tensor<3xi64>) -> tensor<2x?x4xf32>
// CHECK-NOT:       "onnx.Reshape"
// CHECK-NOT:       "onnx.Tile"
// CHECK:           return [[EXPAND_]] : tensor<2x?x4xf32>
// CHECK:         }
}
