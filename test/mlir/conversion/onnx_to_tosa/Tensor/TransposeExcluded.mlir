// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Transpose" -cse %s -split-input-file | FileCheck %s --check-prefix=EXCLUDE

func.func @test_gather_transposes(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// CHECK-LABEL:   func.func @test_gather_transposes(
// CHECK:           tosa.transpose
// CHECK:           tosa.gather
// CHECK:           tosa.transpose

// EXCLUDE-LABEL:   func.func @test_gather_transposes(
// EXCLUDE-NOT:       tosa.transpose
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE:           tosa.gather
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE-NOT:       tosa.transpose
}

// -----

func.func @test_averagepool_transposes(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {kernel_shape = [3, 3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  return %0 : tensor<5x5x30x30xf32>

// CHECK-LABEL:   func.func @test_averagepool_transposes(
// CHECK:           tosa.transpose
// CHECK:           tosa.avg_pool2d
// CHECK:           tosa.transpose

// EXCLUDE-LABEL:   func.func @test_averagepool_transposes(
// EXCLUDE-NOT:       tosa.transpose
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE:           tosa.avg_pool2d
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE-NOT:       tosa.transpose
}

// -----

func.func @test_maxpool_transposes(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3, 3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  return %0 : tensor<5x5x30x30xf32>

// CHECK-LABEL:   func.func @test_maxpool_transposes(
// CHECK:           tosa.transpose
// CHECK:           tosa.max_pool2d
// CHECK:           tosa.transpose

// EXCLUDE-LABEL:   func.func @test_maxpool_transposes(
// EXCLUDE-NOT:       tosa.transpose
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE:           tosa.max_pool2d
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE-NOT:       tosa.transpose
}

// -----

func.func @test_resize_transposes(%arg0 : tensor<1x1x2x4xf32>) -> tensor<1x1x4x8xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "pytorch_half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "linear", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x4xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x8xf32>
  return %0 : tensor<1x1x4x8xf32>

// CHECK-LABEL:   func.func @test_resize_transposes(
// CHECK:           tosa.transpose
// CHECK:           tosa.resize
// CHECK:           tosa.transpose

// EXCLUDE-LABEL:   func.func @test_resize_transposes(
// EXCLUDE-NOT:       tosa.transpose
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE:           tosa.resize
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE-NOT:       tosa.transpose
}

// -----

func.func @test_gemm_transpose_attr(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x4xf32>) -> tensor<3x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transA = 1 : si64} : (tensor<2x3xf32>, tensor<2x4xf32>, none) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>

// CHECK-LABEL:   func.func @test_gemm_transpose_attr(
// CHECK:           tosa.transpose
// CHECK:           tosa.matmul

// EXCLUDE-LABEL:   func.func @test_gemm_transpose_attr(
// EXCLUDE-NOT:       tosa.transpose
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE:           tosa.matmul
// EXCLUDE-NOT:       tosa.transpose
}

// -----

func.func @test_matmul_broadcast_transposes(%arg0 : tensor<3x1x5x6xf32>, %arg1 : tensor<1x4x6x7xf32>) -> tensor<3x4x5x7xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<3x1x5x6xf32>, tensor<1x4x6x7xf32>) -> tensor<3x4x5x7xf32>
  return %0 : tensor<3x4x5x7xf32>

// CHECK-LABEL:   func.func @test_matmul_broadcast_transposes(
// CHECK:           tosa.transpose
// CHECK:           tosa.matmul
// CHECK:           tosa.transpose

// EXCLUDE-LABEL:   func.func @test_matmul_broadcast_transposes(
// EXCLUDE-NOT:       tosa.transpose
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE:           tosa.matmul
// EXCLUDE:           "onnx.Transpose"
// EXCLUDE-NOT:       tosa.transpose
}
