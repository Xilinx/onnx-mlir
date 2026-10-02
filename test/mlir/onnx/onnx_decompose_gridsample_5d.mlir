// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-gridsample-5d-decompose %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// -----

func.func @test_grid_sample_5d_depth_one(
    %input: tensor<1x2x1x3x4xf32>,
    %grid: tensor<1x1x2x3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"
  } : (tensor<1x2x1x3x4xf32>, tensor<1x1x2x3x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL: func.func @test_grid_sample_5d_depth_one
// CHECK: "onnx.GridSample"{{.*}} : (tensor<1x2x3x4xf32>, tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
// CHECK: "onnx.GatherElements"{{.*}} {axis = 2 : si64} : (tensor<1x2x1x2x3xf32>, tensor<1x2x1x2x3xi64>) -> tensor<1x2x1x2x3xf32>
// CHECK: "onnx.Where"
// CHECK: "onnx.GatherElements"{{.*}} {axis = 2 : si64} : (tensor<1x2x1x2x3xf32>, tensor<1x2x1x2x3xi64>) -> tensor<1x2x1x2x3xf32>
// CHECK: "onnx.Where"
// CHECK: return {{.*}} : tensor<1x2x1x2x3xf32>
}

// -----

func.func @test_grid_sample_5d_linear_zeros(
    %input: tensor<2x3x4x5x6xf32>,
    %grid: tensor<2x2x3x5x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"
  } : (tensor<2x3x4x5x6xf32>, tensor<2x2x3x5x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL: func.func @test_grid_sample_5d_linear_zeros
// CHECK: "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3, 4]} : (tensor<2x3x4x5x6xf32>) -> tensor<2x4x3x5x6xf32>
// CHECK: "onnx.Reshape"{{.*}} -> tensor<8x3x5x6xf32>
// CHECK: "onnx.GridSample"{{.*}} {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<8x3x5x6xf32>, tensor<8x3x5x2xf32>) -> tensor<8x3x3x5xf32>
// CHECK: "onnx.GatherElements"{{.*}} {axis = 2 : si64} : (tensor<2x3x4x3x5xf32>, tensor<2x3x1x3x5xi64>) -> tensor<2x3x1x3x5xf32>
// CHECK: "onnx.Where"
// CHECK: "onnx.GatherElements"{{.*}} {axis = 2 : si64} : (tensor<2x3x4x3x5xf32>, tensor<2x3x1x3x5xi64>) -> tensor<2x3x1x3x5xf32>
// CHECK: "onnx.Where"
// CHECK: "onnx.GridSample"{{.*}} {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<8x3x5x6xf32>, tensor<8x3x5x2xf32>) -> tensor<8x3x3x5xf32>
// CHECK: "onnx.GatherElements"{{.*}} {axis = 2 : si64} : (tensor<2x3x4x3x5xf32>, tensor<2x3x1x3x5xi64>) -> tensor<2x3x1x3x5xf32>
// CHECK: "onnx.Where"
// CHECK: "onnx.GatherElements"{{.*}} {axis = 2 : si64} : (tensor<2x3x4x3x5xf32>, tensor<2x3x1x3x5xi64>) -> tensor<2x3x1x3x5xf32>
// CHECK: "onnx.Where"
// CHECK: "onnx.Concat"{{.*}} {axis = 2 : si64} : (tensor<2x3x1x3x5xf32>, tensor<2x3x1x3x5xf32>) -> tensor<2x3x2x3x5xf32>
// CHECK-NOT: "onnx.GridSample"{{.*}}tensor<2x3x4x5x6xf32>
// DISABLED-LABEL: func.func @test_grid_sample_5d_linear_zeros
// DISABLED: "onnx.GridSample"{{.*}} : (tensor<2x3x4x5x6xf32>, tensor<2x2x3x5x3xf32>) -> tensor<2x3x2x3x5xf32>
}

// -----

func.func @test_grid_sample_5d_nearest_unchanged(
    %input: tensor<1x2x3x4x5xf32>,
    %grid: tensor<1x2x3x4x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "nearest", padding_mode = "zeros"
  } : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL: func.func @test_grid_sample_5d_nearest_unchanged
// CHECK: "onnx.GridSample"{{.*}} {align_corners = 0 : si64, mode = "nearest", padding_mode = "zeros"} : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<1x2x2x3x4xf32>
}

// -----

func.func @test_grid_sample_5d_border_unchanged(
    %input: tensor<1x2x3x4x5xf32>,
    %grid: tensor<1x2x3x4x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "linear", padding_mode = "border"
  } : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL: func.func @test_grid_sample_5d_border_unchanged
// CHECK: "onnx.GridSample"{{.*}} {align_corners = 0 : si64, mode = "linear", padding_mode = "border"} : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<1x2x2x3x4xf32>
}

// -----

func.func @test_grid_sample_5d_align_corners_unchanged(
    %input: tensor<1x2x3x4x5xf32>,
    %grid: tensor<1x2x3x4x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 1 : si64, mode = "linear", padding_mode = "zeros"
  } : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL: func.func @test_grid_sample_5d_align_corners_unchanged
// CHECK: "onnx.GridSample"{{.*}} {align_corners = 1 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<1x2x2x3x4xf32>
}

// -----

func.func @test_grid_sample_4d_unchanged(
    %input: tensor<1x2x4x5xf32>,
    %grid: tensor<1x3x4x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"
  } : (tensor<1x2x4x5xf32>, tensor<1x3x4x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL: func.func @test_grid_sample_4d_unchanged
// CHECK: "onnx.GridSample"{{.*}} : (tensor<1x2x4x5xf32>, tensor<1x3x4x2xf32>) -> tensor<1x2x3x4xf32>
}

// -----

func.func @test_grid_sample_5d_dynamic_unchanged(
    %input: tensor<1x2x?x4x5xf32>,
    %grid: tensor<1x2x3x4x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"
  } : (tensor<1x2x?x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL: func.func @test_grid_sample_5d_dynamic_unchanged
// CHECK: "onnx.GridSample"{{.*}} : (tensor<1x2x?x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<1x2x2x3x4xf32>
}

// -----

func.func @test_grid_sample_5d_f16_unchanged(
    %input: tensor<1x2x3x4x5xf16>,
    %grid: tensor<1x2x3x4x3xf16>) -> tensor<*xf16> {
  %0 = "onnx.GridSample"(%input, %grid) {
    align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"
  } : (tensor<1x2x3x4x5xf16>, tensor<1x2x3x4x3xf16>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL: func.func @test_grid_sample_5d_f16_unchanged
// CHECK: "onnx.GridSample"{{.*}} : (tensor<1x2x3x4x5xf16>, tensor<1x2x3x4x3xf16>) -> tensor<1x2x2x3x4xf16>
}
