// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --decompose-onnx=enable-onehot-decompose=true --split-input-file %s | FileCheck %s
// RUN: onnx-mlir-opt --decompose-onnx=enable-onehot-decompose=true --constprop-onnx --split-input-file %s | FileCheck %s --check-prefix=CONSTPROP

// RUN: onnx-mlir-opt --decompose-onnx --split-input-file %s | FileCheck %s --check-prefix=DISABLED
// RUN: onnx-mlir-opt --decompose-onnx=enable-onehot-decompose=false --split-input-file %s | FileCheck %s --check-prefix=DISABLED
// RUN: onnx-mlir-opt --onnx-hybrid-transform="enable-onehot-decompose=true" --split-input-file %s | FileCheck %s --check-prefix=CONSTPROP
// RUN: onnx-mlir-opt --onnx-hybrid-transform --split-input-file %s | FileCheck %s --check-prefix=DISABLED

// -----

func.func @onehot_default_axis_static_depth(
    %indices: tensor<2x3xi64>, %values: tensor<2xf32>) -> tensor<2x3x4xf32> {
  %depth = onnx.Constant dense<4> : tensor<i64>
  %result = "onnx.OneHot"(%indices, %depth, %values) {axis = -1 : si64} :
      (tensor<2x3xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @onehot_default_axis_static_depth
// CHECK-SAME: %[[INDICES:.*]]: tensor<2x3xi64>, %[[VALUES:.*]]: tensor<2xf32>
// CHECK-NOT: "onnx.OneHot"
// CHECK-DAG: %[[DEPTH:.*]] = onnx.Constant dense<4> : tensor<i64>
// CHECK-DAG: %[[ZERO:.*]] = onnx.Constant dense<0> : tensor<i64>
// CHECK-DAG: %[[ONE:.*]] = onnx.Constant dense<1> : tensor<i64>
// CHECK-DAG: %[[AXIS:.*]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG: %[[RANGE_SHAPE:.*]] = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
// CHECK: "onnx.Less"(%[[INDICES]], %[[ZERO]])
// CHECK: "onnx.Add"(%[[INDICES]], %[[DEPTH]])
// CHECK: "onnx.Where"
// CHECK: "onnx.Range"(%[[ZERO]], %[[DEPTH]], %[[ONE]])
// CHECK: "onnx.Unsqueeze"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Equal"
// CHECK: "onnx.Slice"(%[[VALUES]]
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Slice"(%[[VALUES]]
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Where"

// CONSTPROP-LABEL: func.func @onehot_default_axis_static_depth
// CONSTPROP-NOT: "onnx.OneHot"
// CONSTPROP-NOT: "onnx.Range"
// CONSTPROP: "onnx.Equal"
// CONSTPROP: "onnx.Where"

// -----

func.func @onehot_axis_zero_vector_depth(
    %indices: tensor<2x3xi64>, %values: tensor<2xf16>) -> tensor<4x2x3xf16> {
  %depth = onnx.Constant dense<4> : tensor<1xi64>
  %result = "onnx.OneHot"(%indices, %depth, %values) {axis = 0 : si64} :
      (tensor<2x3xi64>, tensor<1xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
  onnx.Return %result : tensor<4x2x3xf16>
}

// CHECK-LABEL: func.func @onehot_axis_zero_vector_depth
// CHECK-NOT: "onnx.OneHot"
// CHECK-DAG: onnx.Constant dense<[4, 1, 1]> : tensor<3xi64>
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Range"
// CHECK: "onnx.Unsqueeze"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Equal"
// CHECK: "onnx.Where"

// CONSTPROP-LABEL: func.func @onehot_axis_zero_vector_depth
// CONSTPROP-NOT: "onnx.OneHot"
// CONSTPROP-NOT: "onnx.Range"
// CONSTPROP: "onnx.Equal"
// CONSTPROP: "onnx.Where"

// -----

func.func @onehot_middle_axis_static_depth(
    %indices: tensor<2x3xi64>, %values: tensor<2xf32>) -> tensor<2x4x3xf32> {
  %depth = onnx.Constant dense<4> : tensor<i64>
  %result = "onnx.OneHot"(%indices, %depth, %values) {axis = 1 : si64} :
      (tensor<2x3xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x4x3xf32>
  onnx.Return %result : tensor<2x4x3xf32>
}

// CHECK-LABEL: func.func @onehot_middle_axis_static_depth
// CHECK-NOT: "onnx.OneHot"
// CHECK-DAG: onnx.Constant dense<[1, 4, 1]> : tensor<3xi64>
// CHECK: "onnx.Unsqueeze"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Equal"
// CHECK: "onnx.Where"

// CONSTPROP-LABEL: func.func @onehot_middle_axis_static_depth
// CONSTPROP-NOT: "onnx.OneHot"
// CONSTPROP-NOT: "onnx.Range"
// CONSTPROP: "onnx.Equal"
// CONSTPROP: "onnx.Where"

// -----

func.func @onehot_middle_axis_dynamic_depth(
    %indices: tensor<2x3xi32>, %depth: tensor<i32>,
    %values: tensor<2xf32>) -> tensor<2x?x3xf32> {
  %result = "onnx.OneHot"(%indices, %depth, %values) {axis = 1 : si64} :
      (tensor<2x3xi32>, tensor<i32>, tensor<2xf32>) -> tensor<2x?x3xf32>
  onnx.Return %result : tensor<2x?x3xf32>
}

// CHECK-LABEL: func.func @onehot_middle_axis_dynamic_depth
// CHECK-SAME: %[[INDICES:.*]]: tensor<2x3xi32>, %[[DEPTH:.*]]: tensor<i32>
// CHECK-NOT: "onnx.OneHot"
// CHECK-DAG: onnx.Constant dense<[1, -1, 1]> : tensor<3xi64>
// CHECK: %[[INDICES_I64:.*]] = "onnx.Cast"(%[[INDICES]]) {{.*to = i64.*}}
// CHECK: %[[DEPTH_I64:.*]] = "onnx.Cast"(%[[DEPTH]]) {{.*to = i64.*}}
// CHECK: "onnx.Less"(%[[INDICES_I64]]
// CHECK: "onnx.Add"(%[[INDICES_I64]], %[[DEPTH_I64]])
// CHECK: "onnx.Range"({{.*}}, %[[DEPTH_I64]], {{.*}})
// CHECK: "onnx.Unsqueeze"
// CHECK: "onnx.Reshape"
// CHECK: "onnx.Equal"
// CHECK: "onnx.Where"

// CONSTPROP-LABEL: func.func @onehot_middle_axis_dynamic_depth
// CONSTPROP-NOT: "onnx.OneHot"
// CONSTPROP: "onnx.Range"

// DISABLED-LABEL: func.func @onehot_default_axis_static_depth
// DISABLED: "onnx.OneHot"

// DISABLED-LABEL: func.func @onehot_axis_zero_vector_depth
// DISABLED: "onnx.OneHot"

// DISABLED-LABEL: func.func @onehot_middle_axis_static_depth
// DISABLED: "onnx.OneHot"

// DISABLED-LABEL: func.func @onehot_middle_axis_dynamic_depth
// DISABLED: "onnx.OneHot"
