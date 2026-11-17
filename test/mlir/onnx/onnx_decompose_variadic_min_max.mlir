// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --decompose-onnx %s --split-input-file | FileCheck %s

func.func @max_3_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_3_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Max"([[R1]], [[A2]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R2]]

// -----

func.func @max_5_2x3xi32(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>, %arg3: tensor<2x3xi32>, %arg4: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @max_5_2x3xi32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A3:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A4:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Max"([[R1]], [[A2]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-DAG: [[R3:%[^ ]+]] = "onnx.Max"([[R2]], [[A3]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-DAG: [[R4:%[^ ]+]] = "onnx.Max"([[R3]], [[A4]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R4]]

// -----

func.func @max_3_unrankedxf32(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @max_3_unrankedxf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<*xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<*xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<*xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Max"([[R1]], [[A2]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R2]]

// -----

func.func @max_3_dynamicx3xf32(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>, %arg2: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<?x3xf32>, tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @max_3_dynamicx3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Max"([[R1]], [[A2]]) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R2]]

// -----

func.func @max_2_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_2_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @max_1_2x3xf32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_1_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[A0]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @max_broadcast_rank(%arg0: tensor<2x1xf32>, %arg1: tensor<1x3xf32>, %arg2: tensor<4x1x1xf32>) -> tensor<4x2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<2x1xf32>, tensor<1x3xf32>, tensor<4x1x1xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @max_broadcast_rank(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x1xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<1x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<4x1x1xf32>
// CHECK-DAG: [[FIRST:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
// CHECK-DAG: [[LAST:%[^ ]+]] = "onnx.Max"([[FIRST]], [[A2]]) : (tensor<2x3xf32>, tensor<4x1x1xf32>) -> tensor<4x2x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[LAST]]

// -----

func.func @max_broadcast_scalar(%arg0: tensor<f32>, %arg1: tensor<1x3xf32>, %arg2: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_broadcast_scalar(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<f32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<1x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x1xf32>
// CHECK-DAG: [[FIRST:%[^ ]+]] = "onnx.Max"([[A0]], [[A1]]) : (tensor<f32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK-DAG: [[LAST:%[^ ]+]] = "onnx.Max"([[FIRST]], [[A2]]) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Max"
// CHECK: return [[LAST]]

// -----

func.func @min_3_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_3_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Min"([[R1]], [[A2]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R2]]

// -----

func.func @min_5_2x3xi32(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>, %arg3: tensor<2x3xi32>, %arg4: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @min_5_2x3xi32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A3:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-SAME: [[A4:%[^ ,)]+]]: tensor<2x3xi32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Min"([[R1]], [[A2]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-DAG: [[R3:%[^ ]+]] = "onnx.Min"([[R2]], [[A3]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-DAG: [[R4:%[^ ]+]] = "onnx.Min"([[R3]], [[A4]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R4]]

// -----

func.func @min_3_unrankedxf32(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func.func @min_3_unrankedxf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<*xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<*xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<*xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Min"([[R1]], [[A2]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R2]]

// -----

func.func @min_3_dynamicx3xf32(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>, %arg2: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2) : (tensor<?x3xf32>, tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  return %0 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @min_3_dynamicx3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<?x3xf32>
// CHECK-DAG: [[R1:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
// CHECK-DAG: [[R2:%[^ ]+]] = "onnx.Min"([[R1]], [[A2]]) : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R2]]

// -----

func.func @min_2_2x3xf32(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_2_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R]]

// -----

func.func @min_1_2x3xf32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_1_2x3xf32(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[A0]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[R]]

// -----

func.func @min_broadcast_rank(%arg0: tensor<2x1xf32>, %arg1: tensor<1x3xf32>, %arg2: tensor<4x1x1xf32>) -> tensor<4x2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2) : (tensor<2x1xf32>, tensor<1x3xf32>, tensor<4x1x1xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @min_broadcast_rank(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<2x1xf32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<1x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<4x1x1xf32>
// CHECK-DAG: [[FIRST:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
// CHECK-DAG: [[LAST:%[^ ]+]] = "onnx.Min"([[FIRST]], [[A2]]) : (tensor<2x3xf32>, tensor<4x1x1xf32>) -> tensor<4x2x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[LAST]]

// -----

func.func @min_broadcast_scalar(%arg0: tensor<f32>, %arg1: tensor<1x3xf32>, %arg2: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Min"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @min_broadcast_scalar(
// CHECK-SAME: [[A0:%[^ ,)]+]]: tensor<f32>
// CHECK-SAME: [[A1:%[^ ,)]+]]: tensor<1x3xf32>
// CHECK-SAME: [[A2:%[^ ,)]+]]: tensor<2x1xf32>
// CHECK-DAG: [[FIRST:%[^ ]+]] = "onnx.Min"([[A0]], [[A1]]) : (tensor<f32>, tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK-DAG: [[LAST:%[^ ]+]] = "onnx.Min"([[FIRST]], [[A2]]) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
// CHECK-NOT: "onnx.Min"
// CHECK: return [[LAST]]

// -----

