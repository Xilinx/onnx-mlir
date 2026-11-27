// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Where" -cse %s | FileCheck %s --check-prefixes=CHECK,EXCLUDED --implicit-check-not=tosa.select
// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not='"onnx.Where"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Where" --convert-onnx-to-tosa -cse %s | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not='"onnx.Where"'

// Selections introduced by decompositions obey the same exclusion as Where.

func.func @leakyrelu(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.LeakyRelu"(%x) {alpha = 0.1 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @leakyrelu(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @elu(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.Elu"(%x) {alpha = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @elu(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @selu(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.Selu"(%x) {alpha = 1.673263 : f32, gamma = 1.050701 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @selu(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @thresholdedrelu(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.ThresholdedRelu"(%x) {alpha = 1.0 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @thresholdedrelu(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @shrink(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.Shrink"(%x) {bias = 0.5 : f32, lambd = 0.5 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @shrink(
// EXCLUDED-COUNT-2: "onnx.Where"
// LOWERED-COUNT-2: tosa.select
// CHECK: return

func.func @round(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.Round"(%x)  : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @round(
// EXCLUDED-COUNT-2: "onnx.Where"
// LOWERED-COUNT-2: tosa.select
// CHECK: return

func.func @cast(%x: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %result = "onnx.Cast"(%x) {to = i32} : (tensor<2x3xf32>) -> tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @cast(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @prelu(%x: tensor<2x3xf32>, %slope: tensor<3xf32>) -> tensor<2x3xf32> {
  %result = "onnx.PRelu"(%x, %slope)  : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @prelu(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @gather(%x: tensor<2x3xf32>, %indices: tensor<4xi64>) -> tensor<4x3xf32> {
  %result = "onnx.Gather"(%x, %indices) {axis = 0 : si64} : (tensor<2x3xf32>, tensor<4xi64>) -> tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}

// CHECK-LABEL: func.func @gather(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return

func.func @where(%cond: tensor<2x1xi1>, %x: tensor<3xf32>, %y: tensor<f32>) -> tensor<2x3xf32> {
  %result = "onnx.Where"(%cond, %x, %y)  : (tensor<2x1xi1>, tensor<3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @where(
// EXCLUDED-COUNT-1: "onnx.Where"
// LOWERED-COUNT-1: tosa.select
// CHECK: return
