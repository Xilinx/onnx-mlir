// RUN: onnx-mlir-opt --mlir-print-debuginfo --shape-inference --convert-onnx-to-tosa=grouped-conv-threshold=4 -cse %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x15x15xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x15x15xf32> loc("Convolution")
  return %0 : tensor<5x2x15x15xf32>
// CHECK-LABEL:   func.func @test_onnx_conv2d_stride_13(

// CHECK-DAG:       tosa.transpose {{.*}}, {{.*}} : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32> loc([[LOC_TRANS:#.+]])
// CHECK-DAG:       tosa.transpose {{.*}}, {{.*}} : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32> loc([[LOC_TRANS:#.+]])
// CHECK:           tosa.conv2d {{.*}}, {{.*}},{{.*}} {dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x245x245x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x15x15x2xf32> loc([[LOC_CONV:#.+]])
// CHECK:           tosa.transpose {{.*}}, {{.*}} : (tensor<5x15x15x2xf32>, tensor<4xi32>) -> tensor<5x2x15x15xf32> loc([[LOC_TRANS:#.+]])
// CHECK-DAG:       [[LOC_TRANS:#.+]] = loc("Compiler generated ONNX to TOSA data format conversion for: 'Convolution'")
// CHECK-DAG:       [[LOC_CONV:#.+]] = loc("Convolution")
}

