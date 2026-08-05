// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Cast" %s | mlir-opt -allow-unregistered-dialect --tosa-layerwise-constant-fold="fold-splat-or-single-use-only=false" --canonicalize | FileCheck %s

// The values exercise ordinary rounding and ties on both sides of zero:
//   2.7 -> 3, 8.5 -> 8, 9.5 -> 10, -1.5 -> -2, -2.5 -> -2, -3.5 -> -4.
// The check is on the floating-point result before the excluded onnx.Cast
// narrows it to i32, so it directly tests TosaBuilder::roundEven.
func.func @test_quantize_linear_round_even_values() -> tensor<6xi8> {
  %input = onnx.Constant dense<[2.7, 8.5, 9.5, -1.5, -2.5, -3.5]> : tensor<6xf32>
  %scale = onnx.Constant dense<1.0> : tensor<f32>
  %zero_point = onnx.Constant dense<1> : tensor<i8>
  %result = "onnx.QuantizeLinear"(%input, %scale, %zero_point)
      {axis = 0 : si64} : (tensor<6xf32>, tensor<f32>, tensor<i8>) -> tensor<6xi8>
  return %result : tensor<6xi8>
}

// CHECK-LABEL: func.func @test_quantize_linear_round_even_values
// CHECK:       "tosa.const"() <{value = dense<[3.000000e+00, 8.000000e+00, 1.000000e+01, -2.000000e+00, -2.000000e+00, -4.000000e+00]> : tensor<6xf32>}> : () -> tensor<6xf32>
