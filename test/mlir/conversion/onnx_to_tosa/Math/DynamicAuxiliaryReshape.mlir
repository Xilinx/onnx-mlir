// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=shape. --implicit-check-not=arith.index_cast --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" -cse %s -split-input-file | FileCheck %s --implicit-check-not=shape. --implicit-check-not='"onnx.Reshape"'
// RUN: onnx-mlir-opt --convert-onnx-to-tosa="excluded-ops=Reshape" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --implicit-check-not=shape. --implicit-check-not='"onnx.Reshape"'

// Keep dynamic computations in ONNX instead of creating auxiliary reshapes.
func.func @gemm_dynamic(%a: tensor<?x5xf32>, %b: tensor<5x?xf32>, %c: tensor<?x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%a, %b, %c) : (tensor<?x5xf32>, tensor<5x?xf32>, tensor<?x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @gemm_dynamic(
// CHECK: %[[RESULT:.*]] = "onnx.Gemm"
// CHECK-NOT: tosa.
// CHECK: return %[[RESULT]]
