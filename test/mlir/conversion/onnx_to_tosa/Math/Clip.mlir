// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_clip_float(%arg0 : tensor<4x4x10x10xf32>) -> tensor<4x4x10x10xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %1 = onnx.Constant dense<10.000000e+00> : tensor<f32>
  %2 = "onnx.Clip"(%arg0, %0, %1) : (tensor<4x4x10x10xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4x10x10xf32>
  "func.return"(%2) : (tensor<4x4x10x10xf32>) -> ()
// CHECK-LABEL:  func.func @test_clip_float
// CHECK-SAME:    ([[ARG_0_:%.+]]: tensor<4x4x10x10xf32>) -> tensor<4x4x10x10xf32> {
// CHECK-NEXT:    [[CLAMP:%.+]] = "tosa.clamp"([[ARG_0_]]) <{max_fp = 1.000000e+01 : f32, max_int = 9223372036854775807 : i64, min_fp = 0.000000e+00 : f32, min_int = -9223372036854775808 : i64}> : (tensor<4x4x10x10xf32>) -> tensor<4x4x10x10xf32>
// CHECK-NEXT:    return [[CLAMP]] : tensor<4x4x10x10xf32>
}

// -----
func.func @test_clip_int(%arg0 : tensor<4x4x10x10xi64>) -> tensor<4x4x10x10xi64> {
  %0 = onnx.Constant dense<2> : tensor<i64>
  %1 = onnx.Constant dense<24> : tensor<i64>
  %2 = "onnx.Clip"(%arg0, %0, %1) : (tensor<4x4x10x10xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4x10x10xi64>
  "func.return"(%2) : (tensor<4x4x10x10xi64>) -> ()
// CHECK-LABEL:  func.func @test_clip_int
// CHECK-SAME:    ([[ARG:%.+]]: tensor<4x4x10x10xi64>) -> tensor<4x4x10x10xi64> {
// CHECK-NEXT:    [[CLAMP:%.+]] = "tosa.clamp"([[ARG]]) <{max_fp = 3.40282347E+38 : f32, max_int = 24 : i64, min_fp = -3.40282347E+38 : f32, min_int = 2 : i64}> : (tensor<4x4x10x10xi64>) -> tensor<4x4x10x10xi64>
// CHECK-NEXT:    return [[CLAMP]] : tensor<4x4x10x10xi64>
}


