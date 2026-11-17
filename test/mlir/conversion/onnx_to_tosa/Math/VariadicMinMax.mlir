// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --decompose-onnx --shape-inference --convert-onnx-to-tosa %s --split-input-file | FileCheck %s --implicit-check-not='"onnx.'

func.func @max_variadic(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_variadic(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = tosa.maximum [[A0]], [[A1]]
// CHECK-DAG: [[R2:%[^ ]+]] = tosa.maximum [[R1]], [[A2]]
// CHECK-NOT: tosa.maximum
// CHECK: return [[R2]]

// -----

func.func @min_variadic(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>, %arg3: tensor<2x3xf32>, %arg4: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_variadic(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A3:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A4:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = tosa.minimum [[A0]], [[A1]]
// CHECK-DAG: [[R2:%[^ ]+]] = tosa.minimum [[R1]], [[A2]]
// CHECK-DAG: [[R3:%[^ ]+]] = tosa.minimum [[R2]], [[A3]]
// CHECK-DAG: [[R4:%[^ ]+]] = tosa.minimum [[R3]], [[A4]]
// CHECK-NOT: tosa.minimum
// CHECK: return [[R4]]

// -----

