// RUN: onnx-mlir-opt --split-input-file %s -constprop-onnx=enable-qdq | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive cases.
//===----------------------------------------------------------------------===//

// Per-tensor, no zero point. 1.0/0.5=2, 2.0/0.5=4, 3.0/0.5=6.
func.func @fold_q_per_tensor() -> tensor<3xui8> {
  %x = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<3xf32>, tensor<f32>, tensor<ui8>) -> tensor<3xui8>
  return %q : tensor<3xui8>
}

// CHECK-LABEL: @fold_q_per_tensor
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[2, 4, 6]> : tensor<3xui8>

// -----

// Per-tensor with a non-zero zero point. round(4.0/0.5)+10 = 8+10 = 18.
func.func @fold_q_with_zp() -> tensor<2xui8> {
  %x = onnx.Constant dense<[4.0, 0.0]> : tensor<2xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  return %q : tensor<2xui8>
}

// CHECK-LABEL: @fold_q_with_zp
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[18, 10]> : tensor<2xui8>

// -----

// Round to nearest, ties to even: 0.5->0, 1.5->2, 2.5->2, 3.5->4 with scale 1.
func.func @fold_q_round_ties_even() -> tensor<4xi8> {
  %x = onnx.Constant dense<[0.5, 1.5, 2.5, 3.5]> : tensor<4xf32>
  %scale = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>
  return %q : tensor<4xi8>
}

// CHECK-LABEL: @fold_q_round_ties_even
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[0, 2, 2, 4]> : tensor<4xi8>

// -----

// Saturation: 1000/1=1000 clips to 255 (ui8 max), -5/1 clips to 0 (ui8 min).
func.func @fold_q_saturate() -> tensor<2xui8> {
  %x = onnx.Constant dense<[1000.0, -5.0]> : tensor<2xf32>
  %scale = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  return %q : tensor<2xui8>
}

// CHECK-LABEL: @fold_q_saturate
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[255, 0]> : tensor<2xui8>

// -----

// Per-axis along axis 0: row 0 uses scale 0.5, row 1 uses scale 0.25.
// row0: [2.0/0.5, 4.0/0.5] = [4, 8]; row1: [1.0/0.25, 2.0/0.25] = [4, 8].
func.func @fold_q_per_axis() -> tensor<2x2xi8> {
  %x = onnx.Constant dense<[[2.0, 4.0], [1.0, 2.0]]> : tensor<2x2xf32>
  %scale = onnx.Constant dense<[5.000000e-01, 2.500000e-01]> : tensor<2xf32>
  %zp = onnx.Constant dense<[0, 0]> : tensor<2xi8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xi8>) -> tensor<2x2xi8>
  return %q : tensor<2x2xi8>
}

// CHECK-LABEL: @fold_q_per_axis
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<{{\[}}[4, 8], [4, 8]]> : tensor<2x2xi8>

// -----

// No zero point operand (None). round(2.0/0.5) = 4.
func.func @fold_q_no_zp() -> tensor<1xi8> {
  %x = onnx.Constant dense<[2.0]> : tensor<1xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %q = "onnx.QuantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1xf32>, tensor<f32>, none) -> tensor<1xi8>
  return %q : tensor<1xi8>
}

// CHECK-LABEL: @fold_q_no_zp
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<4> : tensor<1xi8>

// -----

// End-to-end: Const(fp) -> Q -> DQ collapses to Const(int) -> DQ.
func.func @fold_q_then_dq() -> tensor<3xf32> {
  %x = onnx.Constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<3xf32>, tensor<f32>, tensor<ui8>) -> tensor<3xui8>
  %dq = "onnx.DequantizeLinear"(%q, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<3xui8>, tensor<f32>, tensor<ui8>) -> tensor<3xf32>
  return %dq : tensor<3xf32>
}

// CHECK-LABEL: @fold_q_then_dq
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[2, 4, 6]> : tensor<3xui8>
// CHECK: onnx.DequantizeLinear

//===----------------------------------------------------------------------===//
// Negative cases.
//===----------------------------------------------------------------------===//

// -----

// Input is not a constant (function argument): Q must remain.
func.func @no_fold_non_const(%arg0: tensor<3xf32>) -> tensor<3xui8> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %q = "onnx.QuantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<3xf32>, tensor<f32>, tensor<ui8>) -> tensor<3xui8>
  return %q : tensor<3xui8>
}

// CHECK-LABEL: @no_fold_non_const
// CHECK: onnx.QuantizeLinear

// -----

// Blocked quantization (block_size != 0): Q must remain.
func.func @no_fold_blocked() -> tensor<4xui8> {
  %x = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %scale = onnx.Constant dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>
  %zp = onnx.Constant dense<[0, 0]> : tensor<2xui8>
  %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<4xf32>, tensor<2xf32>, tensor<2xui8>) -> tensor<4xui8>
  return %q : tensor<4xui8>
}

// CHECK-LABEL: @no_fold_blocked
// CHECK: onnx.QuantizeLinear
