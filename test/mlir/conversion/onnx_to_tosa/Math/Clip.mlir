// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_clip_float(%arg0 : tensor<4x4x10x10xf32>) -> tensor<4x4x10x10xf32> {
  %0 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %1 = onnx.Constant dense<10.000000e+00> : tensor<f32>
  %2 = "onnx.Clip"(%arg0, %0, %1) : (tensor<4x4x10x10xf32>, tensor<f32>, tensor<f32>) -> tensor<4x4x10x10xf32>
  "func.return"(%2) : (tensor<4x4x10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.clamp"([[PARAM_0_]]) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}