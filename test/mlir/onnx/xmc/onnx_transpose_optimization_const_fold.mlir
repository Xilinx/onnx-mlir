// RUN: onnx-mlir-opt --onnx-transpose-optimization --constprop-onnx %s | FileCheck %s

// Transpose pushed through a binary op with a constant operand: the inserted
// Transpose(const) folds into a transposed constant of the same payload type.

// Test: Splat f32 payload behind a quantized result type.
// CHECK-LABEL: func @test_push_transpose_through_sub_quant_const_f32_splat
func.func @test_push_transpose_through_sub_quant_const_f32_splat(%arg0: tensor<1x4x4x8x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 3.125000e-02>> {
  // CHECK: %[[C:.*]] = onnx.Constant {value = dense<5.000000e-01> : tensor<1x4x4x8xf32>} : tensor<1x4x4x8x!quant.uniform<i8:f32, 7.812500e-03>>
  // CHECK: %[[S:.*]] = "onnx.Sub"(%arg0, %[[C]]) {{.*}} -> tensor<1x4x4x8x!quant.uniform<i8:f32, 3.125000e-02>>
  // CHECK: "onnx.Transpose"(%[[S]]) {perm = [0, 3, 1, 2]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<1x4x4x8x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = onnx.Constant {value = dense<5.000000e-01> : tensor<1x8x4x4xf32>} : tensor<1x8x4x4x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "onnx.Sub"(%0, %1) : (tensor<1x8x4x4x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<1x8x4x4x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x8x4x4x!quant.uniform<i8:f32, 3.125000e-02>>
  return %2 : tensor<1x8x4x4x!quant.uniform<i8:f32, 3.125000e-02>>
}

// -----
// Test: Non-splat f32 payload behind a quantized result type.
// CHECK-LABEL: func @test_push_transpose_through_sub_quant_const_f32_nonsplat
func.func @test_push_transpose_through_sub_quant_const_f32_nonsplat(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x3x2x!quant.uniform<i8:f32, 3.125000e-02>> {
  // CHECK: %[[C:.*]] = onnx.Constant {value = dense<{{\[\[\[}}1.000000e+00, 3.000000e+00, 5.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]]> : tensor<1x2x3xf32>} : tensor<1x2x3x!quant.uniform<i8:f32, 7.812500e-03>>
  // CHECK: %[[S:.*]] = "onnx.Sub"(%arg0, %[[C]]) {{.*}} -> tensor<1x2x3x!quant.uniform<i8:f32, 3.125000e-02>>
  // CHECK: "onnx.Transpose"(%[[S]]) {perm = [0, 2, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x2x3x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = onnx.Constant {value = dense<[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]> : tensor<1x3x2xf32>} : tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "onnx.Sub"(%0, %1) : (tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x3x2x!quant.uniform<i8:f32, 3.125000e-02>>
  return %2 : tensor<1x3x2x!quant.uniform<i8:f32, 3.125000e-02>>
}

// -----
// Test: Non-splat f16 payload.
// CHECK-LABEL: func @test_push_transpose_through_add_const_f16_nonsplat
func.func @test_push_transpose_through_add_const_f16_nonsplat(%arg0: tensor<1x2x3xf16>) -> tensor<1x3x2xf16> {
  // CHECK: %[[C:.*]] = onnx.Constant dense<{{\[\[\[}}1.000000e+00, 3.000000e+00, 5.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]]> : tensor<1x2x3xf16>
  // CHECK: %[[A:.*]] = "onnx.Add"(%arg0, %[[C]]) {{.*}} -> tensor<1x2x3xf16>
  // CHECK: "onnx.Transpose"(%[[A]]) {perm = [0, 2, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x2x3xf16>) -> tensor<1x3x2xf16>
  %1 = onnx.Constant {value = dense<[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]> : tensor<1x3x2xf16>} : tensor<1x3x2xf16>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x3x2xf16>, tensor<1x3x2xf16>) -> tensor<1x3x2xf16>
  return %2 : tensor<1x3x2xf16>
}

// -----
// Test: Non-splat i8 storage payload.
// CHECK-LABEL: func @test_push_transpose_through_sub_quant_const_i8_nonsplat
func.func @test_push_transpose_through_sub_quant_const_i8_nonsplat(%arg0: tensor<1x2x3x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x3x2x!quant.uniform<i8:f32, 3.125000e-02>> {
  // CHECK: %[[C:.*]] = onnx.Constant {value = dense<{{\[\[\[}}1, 3, -5], [2, 4, -6]]]> : tensor<1x2x3xi8>} : tensor<1x2x3x!quant.uniform<i8:f32, 7.812500e-03>>
  // CHECK: "onnx.Sub"(%arg0, %[[C]])
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x2x3x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = onnx.Constant {value = dense<[[[1, 2], [3, 4], [-5, -6]]]> : tensor<1x3x2xi8>} : tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "onnx.Sub"(%0, %1) : (tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<1x3x2x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x3x2x!quant.uniform<i8:f32, 3.125000e-02>>
  return %2 : tensor<1x3x2x!quant.uniform<i8:f32, 3.125000e-02>>
}

// -----
// Test: Lower-rank constant is expanded with leading 1s before the transpose.
// CHECK-LABEL: func @test_push_transpose_through_mul_const_rank_expand
func.func @test_push_transpose_through_mul_const_rank_expand(%arg0: tensor<1x2x3xf32>) -> tensor<1x3x2xf32> {
  // CHECK: %[[C:.*]] = onnx.Constant dense<{{\[\[\[}}1.000000e+00, 3.000000e+00, 5.000000e+00], [2.000000e+00, 4.000000e+00, 6.000000e+00]]]> : tensor<1x2x3xf32>
  // CHECK: %[[M:.*]] = "onnx.Mul"(%arg0, %[[C]]) {{.*}} -> tensor<1x2x3xf32>
  // CHECK: "onnx.Transpose"(%[[M]]) {perm = [0, 2, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x2x3xf32>) -> tensor<1x3x2xf32>
  %1 = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  %2 = "onnx.Mul"(%0, %1) : (tensor<1x3x2xf32>, tensor<3x2xf32>) -> tensor<1x3x2xf32>
  return %2 : tensor<1x3x2xf32>
}
