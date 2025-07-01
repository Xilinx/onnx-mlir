// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<1x10x10x1xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<1x10x10x1xf32>
  func.return %1 : tensor<1x10x10x1xf32>
// CHECK-LABEL:  func.func @test_unsqueeze
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<1x10x10x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 10, 10, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<10x10xf32>, !tosa.shape<4>) -> tensor<1x10x10x1xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x10x10x1xf32>
// CHECK:         }
}

func.func @test_unsqueeze_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> tensor<16x32x1x64xf32>
  func.return %1 : tensor<16x32x1x64xf32>
// CHECK-LABEL:  func.func @test_unsqueeze_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[16, 32, 1, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<16x32x64xf32>, !tosa.shape<4>) -> tensor<16x32x1x64xf32>
// CHECK:           return [[VAR_1_]] : tensor<16x32x1x64xf32>
// CHECK:         }
}

func.func @test_unsqueeze_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<2xi64>) -> tensor<16x1x32x1x64xf32>
  func.return %1 : tensor<16x1x32x1x64xf32>
// CHECK-LABEL:  func.func @test_unsqueeze_mix
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[16, 1, 32, 1, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<16x32x64xf32>, !tosa.shape<5>) -> tensor<16x1x32x1x64xf32>
// CHECK:           return [[VAR_1_]] : tensor<16x1x32x1x64xf32>
// CHECK:         }
}

// -----

func.func @unsqueeze_runtime(%arg0: tensor<3x4x5xf32> , %arg1: tensor<1xi64> ) -> tensor<3x4x1x5xf32> {
  %0 = "onnx.Unsqueeze"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<1xi64>) -> tensor<3x4x1x5xf32>
  return %0 : tensor<3x4x1x5xf32>
// CHECK-LABEL:  func.func @unsqueeze_runtime
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<3x4x1x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[3, 4, 1, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<3x4x5xf32>, !tosa.shape<4>) -> tensor<3x4x1x5xf32>
// CHECK:           return [[VAR_1_]] : tensor<3x4x1x5xf32>
// CHECK:         }
}
// -----

func.func @unsqueeze_dynamic(%arg0: tensor<1x3x4x5xf32> , %arg1: tensor<1xi64> ) -> tensor<?x?x?xf32> {
  %0 = "onnx.Unsqueeze"(%arg0, %arg1) : (tensor<1x3x4x5xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
// CHECK-LABEL: unsqueeze_dynamic
// CHECK: onnx.Unsqueeze
}
