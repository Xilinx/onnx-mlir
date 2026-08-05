// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Identity elimination.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cast_identity_elimination
func.func @cast_identity_elimination(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
  // CHECK: return %arg0
}

// -----

//===----------------------------------------------------------------------===//
/// Integer cast-chain folding.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @integer_cast_chain_narrowing
func.func @integer_cast_chain_narrowing(%arg0: tensor<3xi32>) -> tensor<3xi8> {
  %0 = "onnx.Cast"(%arg0) {to = i16} : (tensor<3xi32>) -> tensor<3xi16>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<3xi16>) -> tensor<3xi8>
  return %1 : tensor<3xi8>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i8}
  // CHECK-NOT: {to = i16}
}

// -----

// CHECK-LABEL: @unsigned_integer_cast_chain_narrowing
func.func @unsigned_integer_cast_chain_narrowing(%arg0: tensor<3xui32>) -> tensor<3xui8> {
  %0 = "onnx.Cast"(%arg0) {to = ui16} : (tensor<3xui32>) -> tensor<3xui16>
  %1 = "onnx.Cast"(%0) {to = ui8} : (tensor<3xui16>) -> tensor<3xui8>
  return %1 : tensor<3xui8>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = ui8}
  // CHECK-NOT: {to = ui16}
}

// -----

// CHECK-LABEL: @integer_cast_chain_full_elimination
func.func @integer_cast_chain_full_elimination(%arg0: tensor<3xi8>) -> tensor<3xi8> {
  %0 = "onnx.Cast"(%arg0) {to = i16} : (tensor<3xi8>) -> tensor<3xi16>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<3xi16>) -> tensor<3xi8>
  return %1 : tensor<3xi8>
  // CHECK: return %arg0
  // CHECK-NOT: "onnx.Cast"
}

// -----

// CHECK-LABEL: @integer_cast_chain_widening
func.func @integer_cast_chain_widening(%arg0: tensor<3xi8>) -> tensor<3xi32> {
  %0 = "onnx.Cast"(%arg0) {to = i16} : (tensor<3xi8>) -> tensor<3xi16>
  %1 = "onnx.Cast"(%0) {to = i32} : (tensor<3xi16>) -> tensor<3xi32>
  return %1 : tensor<3xi32>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i32}
  // CHECK-NOT: {to = i16}
}

// -----

// CHECK-LABEL: @unsigned_integer_cast_chain_widening
func.func @unsigned_integer_cast_chain_widening(%arg0: tensor<3xui8>) -> tensor<3xui32> {
  %0 = "onnx.Cast"(%arg0) {to = ui16} : (tensor<3xui8>) -> tensor<3xui16>
  %1 = "onnx.Cast"(%0) {to = ui32} : (tensor<3xui16>) -> tensor<3xui32>
  return %1 : tensor<3xui32>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = ui32}
  // CHECK-NOT: {to = ui16}
}

// -----

// CHECK-LABEL: @integer_cast_chain_final_signedness_change
func.func @integer_cast_chain_final_signedness_change(%arg0: tensor<3xui8>) -> tensor<3xi8> {
  %0 = "onnx.Cast"(%arg0) {to = ui16} : (tensor<3xui8>) -> tensor<3xui16>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<3xui16>) -> tensor<3xi8>
  return %1 : tensor<3xi8>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i8}
  // CHECK-NOT: {to = ui16}
}

// -----

// CHECK-LABEL: @float_cast_chain_widening
func.func @float_cast_chain_widening(%arg0: tensor<3xf16>) -> tensor<3xf64> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xf16>) -> tensor<3xf32>
  %1 = "onnx.Cast"(%0) {to = f64} : (tensor<3xf32>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f64}
  // CHECK-NOT: {to = f32}
}

// -----

// CHECK-LABEL: @fp_mixed_narrow_types
func.func @fp_mixed_narrow_types(%arg0: tensor<1x3x3xf32>) -> tensor<1x3x3xbf16> {
  %0 = "onnx.Cast"(%arg0) {to = f16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xf16>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<1x3x3xf16>) -> tensor<1x3x3xf32>
  %2 = "onnx.Cast"(%1) {to = bf16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xbf16>
  return %2 : tensor<1x3x3xbf16>
  // CHECK: %[[NARROW:.*]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f16}
  // CHECK: %[[FINAL:.*]] = "onnx.Cast"(%[[NARROW]]) {saturate = 1 : si64, to = bf16}
  // CHECK-NOT: {to = f32}
  // CHECK: return %[[FINAL]]
}

// -----

// CHECK-LABEL: @integer_to_float_cast_chain
func.func @integer_to_float_cast_chain(%arg0: tensor<3xi8>) -> tensor<3xi32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xi8>) -> tensor<3xf32>
  %1 = "onnx.Cast"(%0) {to = i32} : (tensor<3xf32>) -> tensor<3xi32>
  return %1 : tensor<3xi32>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i32}
  // CHECK-NOT: {to = f32}
}

// -----

//===----------------------------------------------------------------------===//
/// Lossless f16/bf16 round-trip elimination.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bf16_roundtrip
func.func @bf16_roundtrip(%arg0: tensor<1x3x3xf32>) -> tensor<1x3x3xbf16> {
  %0 = "onnx.Cast"(%arg0) {to = bf16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xbf16>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<1x3x3xbf16>) -> tensor<1x3x3xf32>
  %2 = "onnx.Cast"(%1) {to = bf16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xbf16>
  return %2 : tensor<1x3x3xbf16>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16}
  // CHECK-NOT: {to = f32}
}

// -----

// CHECK-LABEL: @f16_roundtrip
func.func @f16_roundtrip(%arg0: tensor<1x3x3xf32>) -> tensor<1x3x3xf16> {
  %0 = "onnx.Cast"(%arg0) {to = f16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xf16>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<1x3x3xf16>) -> tensor<1x3x3xf32>
  %2 = "onnx.Cast"(%1) {to = f16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xf16>
  return %2 : tensor<1x3x3xf16>
  // CHECK: "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f16}
  // CHECK-NOT: {to = f32}
}

// -----

//===----------------------------------------------------------------------===//
/// Negative cases: patterns must not apply.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @integer_cast_chain_information_losing
func.func @integer_cast_chain_information_losing(%arg0: tensor<3xi32>) -> tensor<3xi16> {
  %0 = "onnx.Cast"(%arg0) {to = i8} : (tensor<3xi32>) -> tensor<3xi8>
  %1 = "onnx.Cast"(%0) {to = i16} : (tensor<3xi8>) -> tensor<3xi16>
  return %1 : tensor<3xi16>
  // CHECK: "onnx.Cast"
  // CHECK: "onnx.Cast"
}

// -----

// CHECK-LABEL: @integer_cast_chain_signedness_mismatch
func.func @integer_cast_chain_signedness_mismatch(%arg0: tensor<3xi8>) -> tensor<3xi8> {
  %0 = "onnx.Cast"(%arg0) {to = ui16} : (tensor<3xi8>) -> tensor<3xui16>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<3xui16>) -> tensor<3xi8>
  return %1 : tensor<3xi8>
  // CHECK: "onnx.Cast"
  // CHECK: "onnx.Cast"
}

// -----

// CHECK-LABEL: @cast_chain_intermediate_multiple_uses
func.func @cast_chain_intermediate_multiple_uses(%arg0: tensor<3xi8>) -> (tensor<3xi32>, tensor<3xf32>) {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<3xi8>) -> tensor<3xf32>
  %1 = "onnx.Cast"(%0) {to = i32} : (tensor<3xf32>) -> tensor<3xi32>
  return %1, %0 : tensor<3xi32>, tensor<3xf32>
  // CHECK: %[[MID:.*]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32}
  // CHECK: %[[FINAL:.*]] = "onnx.Cast"(%[[MID]]) {saturate = 1 : si64, to = i32}
  // CHECK: return %[[FINAL]], %[[MID]]
}

// -----

// CHECK-LABEL: @fp_lossy_roundtrip
func.func @fp_lossy_roundtrip(%arg0: tensor<1x3x3xf32>) -> tensor<1x3x3xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f16} : (tensor<1x3x3xf32>) -> tensor<1x3x3xf16>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<1x3x3xf16>) -> tensor<1x3x3xf32>
  return %1 : tensor<1x3x3xf32>
  // CHECK: "onnx.Cast"
  // CHECK: "onnx.Cast"
}
