// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --recompose-onnx="enable-clip-from-where-min-max=true" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --recompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DEFAULT-OFF

// -----

// Canonical clamp idiom: Where(Greater(x, hi), hi, Where(Less(x, lo), lo, x))
// recomposes into a single onnx.Clip(x, lo, hi).
func.func @clamp_canonical(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %lo = onnx.Constant dense<-1.000000e+00> : tensor<f32>
  %hi = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %ltLo = "onnx.Less"(%x, %lo) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  %lower = "onnx.Where"(%ltLo, %lo, %x) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %gtHi = "onnx.Greater"(%x, %hi) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  %clamped = "onnx.Where"(%gtHi, %hi, %lower) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %clamped : tensor<2x3xf32>

// CHECK-LABEL:  func.func @clamp_canonical
// CHECK-SAME:   ([[X_:%.+]]: tensor<2x3xf32>) -> tensor<2x3xf32> {
// CHECK-DAG:       [[LO_:%.+]] = onnx.Constant dense<-1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[HI_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[CLIP_:%.+]] = "onnx.Clip"([[X_]], [[LO_]], [[HI_]]) : (tensor<2x3xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:           return [[CLIP_]] : tensor<2x3xf32>
// CHECK:         }

// DEFAULT-OFF-LABEL:  func.func @clamp_canonical
// DEFAULT-OFF-NOT:      "onnx.Clip"
// DEFAULT-OFF:          "onnx.Where"
}

// -----

// Nesting-order swapped: outer Where is the lower clamp, inner is the upper.
func.func @clamp_swapped_nesting(%x: tensor<4xf32>) -> tensor<4xf32> {
  %lo = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %gtHi = "onnx.Greater"(%x, %hi) : (tensor<4xf32>, tensor<f32>) -> tensor<4xi1>
  %upper = "onnx.Where"(%gtHi, %hi, %x) : (tensor<4xi1>, tensor<f32>, tensor<4xf32>) -> tensor<4xf32>
  %ltLo = "onnx.Less"(%x, %lo) : (tensor<4xf32>, tensor<f32>) -> tensor<4xi1>
  %clamped = "onnx.Where"(%ltLo, %lo, %upper) : (tensor<4xi1>, tensor<f32>, tensor<4xf32>) -> tensor<4xf32>
  return %clamped : tensor<4xf32>

// CHECK-LABEL:  func.func @clamp_swapped_nesting
// CHECK-SAME:   ([[X_:%.+]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG:       [[LO_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[HI_:%.+]] = onnx.Constant dense<6.000000e+00> : tensor<f32>
// CHECK:           [[CLIP_:%.+]] = "onnx.Clip"([[X_]], [[LO_]], [[HI_]]) : (tensor<4xf32>, tensor<f32>, tensor<f32>) -> tensor<4xf32>
// CHECK:           return [[CLIP_]] : tensor<4xf32>
// CHECK:         }
}

// -----

// OrEqual + reversed compare operands: LessOrEqual(hi, x) == (x >= hi) and
// GreaterOrEqual(lo, x) == (x <= lo). Both spellings still form the clamp.
func.func @clamp_orequal_reversed(%x: tensor<8xf32>) -> tensor<8xf32> {
  %lo = onnx.Constant dense<-2.000000e+00> : tensor<f32>
  %hi = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %geLo = "onnx.GreaterOrEqual"(%lo, %x) : (tensor<f32>, tensor<8xf32>) -> tensor<8xi1>
  %lower = "onnx.Where"(%geLo, %lo, %x) : (tensor<8xi1>, tensor<f32>, tensor<8xf32>) -> tensor<8xf32>
  %leHi = "onnx.LessOrEqual"(%hi, %x) : (tensor<f32>, tensor<8xf32>) -> tensor<8xi1>
  %clamped = "onnx.Where"(%leHi, %hi, %lower) : (tensor<8xi1>, tensor<f32>, tensor<8xf32>) -> tensor<8xf32>
  return %clamped : tensor<8xf32>

// CHECK-LABEL:  func.func @clamp_orequal_reversed
// CHECK:           [[CLIP_:%.+]] = "onnx.Clip"
// CHECK:           return [[CLIP_]] : tensor<8xf32>
// CHECK:         }
}

// -----

// Inverted bounds (lo > hi) fall into Clip's implementation-defined region and
// must NOT be recomposed.
func.func @clamp_inverted_bounds_skipped(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %lo = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %hi = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %ltLo = "onnx.Less"(%x, %lo) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  %lower = "onnx.Where"(%ltLo, %lo, %x) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %gtHi = "onnx.Greater"(%x, %hi) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  %clamped = "onnx.Where"(%gtHi, %hi, %lower) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %clamped : tensor<2x3xf32>

// CHECK-LABEL:  func.func @clamp_inverted_bounds_skipped
// CHECK-NOT:       "onnx.Clip"
// CHECK:           "onnx.Where"
// CHECK:         }
}

// -----

// Non-constant bound cannot be validated at compile time and is left untouched.
func.func @clamp_non_constant_bound_skipped(%x: tensor<2x3xf32>, %hi: tensor<f32>) -> tensor<2x3xf32> {
  %lo = onnx.Constant dense<-1.000000e+00> : tensor<f32>
  %ltLo = "onnx.Less"(%x, %lo) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  %lower = "onnx.Where"(%ltLo, %lo, %x) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %gtHi = "onnx.Greater"(%x, %hi) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  %clamped = "onnx.Where"(%gtHi, %hi, %lower) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %clamped : tensor<2x3xf32>

// CHECK-LABEL:  func.func @clamp_non_constant_bound_skipped
// CHECK-NOT:       "onnx.Clip"
// CHECK:           "onnx.Where"
// CHECK:         }
}
