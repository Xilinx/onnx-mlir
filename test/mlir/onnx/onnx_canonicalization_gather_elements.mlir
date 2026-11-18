// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --enable-expand-canonicalization --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --enable-expand-canonicalization --enable-gather-elements-tile-canonicalization=false --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s --check-prefix=DISABLED

func.func @fuse_gather_elements_tile(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<2x3x4xf32> {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @fuse_gather_elements_tile(
// CHECK-NOT: "onnx.Tile"
// CHECK: [[RESHAPED:%.+]] = "onnx.Reshape"(%arg1, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-NEXT: [[GATHER:%.+]] = "onnx.Gather"(%arg0, [[RESHAPED]]) {axis = 1 : si64} : (tensor<2x5x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NOT: "onnx.GatherElements"
// CHECK-NEXT: onnx.Return [[GATHER]] : tensor<2x3x4xf32>
// DISABLED-LABEL: func.func @fuse_gather_elements_tile(
// DISABLED: "onnx.Tile"
// DISABLED: "onnx.GatherElements"

// -----

func.func @fuse_gather_elements_tile_negative_axis(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<2x3x4xf32> {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = -2 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @fuse_gather_elements_tile_negative_axis(
// CHECK-NOT: "onnx.Tile"
// CHECK: [[RESHAPED:%.+]] = "onnx.Reshape"(%arg1, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-NEXT: [[GATHER:%.+]] = "onnx.Gather"(%arg0, [[RESHAPED]]) {axis = -2 : si64} : (tensor<2x5x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NOT: "onnx.GatherElements"
// CHECK-NEXT: onnx.Return [[GATHER]] : tensor<2x3x4xf32>

// -----

// Repeating the GatherElements axis cannot be represented by a 1-D Gather.
func.func @do_not_fuse_repeated_axis(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<2x6x4xf32> {
  %repeats = onnx.Constant dense<[2, 2, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x6x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x6x4xi64>) -> tensor<2x6x4xf32>
  onnx.Return %result : tensor<2x6x4xf32>
}
// CHECK-LABEL: func.func @do_not_fuse_repeated_axis(
// CHECK: [[TILED:%.+]] = "onnx.Tile"
// CHECK: [[RESULT:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]]) {axis = 1 : si64}
// CHECK: onnx.Return [[RESULT]]

// -----

// Both consumers select rows using the same compact runtime indices.
func.func @fuse_shared_tile_negative_axis(
    %data: tensor<2x5x4xf32>, %other: tensor<2x7x4xf32>,
    %indices: tensor<1x3x1xi32>)
    -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>) {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi32>, tensor<3xi64>) -> tensor<2x3x4xi32>
  %first = "onnx.GatherElements"(%data, %tiled) {axis = -2 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi32>) -> tensor<2x3x4xf32>
  %second = "onnx.GatherElements"(%other, %tiled) {axis = -2 : si64} :
      (tensor<2x7x4xf32>, tensor<2x3x4xi32>) -> tensor<2x3x4xf32>
  onnx.Return %first, %second : tensor<2x3x4xf32>, tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @fuse_shared_tile_negative_axis(
// CHECK-NOT: "onnx.Tile"
// CHECK-NOT: "onnx.GatherElements"
// CHECK: [[FLAT:%.+]] = "onnx.Reshape"(%arg2, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi32>, tensor<1xi64>) -> tensor<3xi32>
// CHECK-NEXT: [[FIRST:%.+]] = "onnx.Gather"(%arg0, [[FLAT]]) {axis = -2 : si64} : (tensor<2x5x4xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
// CHECK-NEXT: [[OTHER_FLAT:%.+]] = "onnx.Reshape"(%arg2, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi32>, tensor<1xi64>) -> tensor<3xi32>
// CHECK-NEXT: [[SECOND:%.+]] = "onnx.Gather"(%arg1, [[OTHER_FLAT]]) {axis = -2 : si64} : (tensor<2x7x4xf32>, tensor<3xi32>) -> tensor<2x3x4xf32>
// CHECK-NEXT: onnx.Return [[FIRST]], [[SECOND]]
// DISABLED-LABEL: func.func @fuse_shared_tile_negative_axis(
// DISABLED: [[TILED:%.+]] = "onnx.Tile"(%arg2,
// DISABLED-DAG: [[FIRST:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]]) {axis = -2 : si64}
// DISABLED-DAG: [[SECOND:%.+]] = "onnx.GatherElements"(%arg1, [[TILED]]) {axis = -2 : si64}
// DISABLED: onnx.Return [[FIRST]], [[SECOND]]

// -----

// Returning the broadcast indices keeps the Tile live after fusion.
func.func @fuse_tile_with_unrelated_use(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> (tensor<2x3x4xf32>, tensor<2x3x4xi64>) {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %result, %tiled : tensor<2x3x4xf32>, tensor<2x3x4xi64>
}
// CHECK-LABEL: func.func @fuse_tile_with_unrelated_use(
// CHECK-DAG: [[TILED:%.+]] = "onnx.Tile"(%arg1,
// CHECK-DAG: [[FLAT:%.+]] = "onnx.Reshape"(%arg1, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK: [[RESULT:%.+]] = "onnx.Gather"(%arg0, [[FLAT]]) {axis = 1 : si64} : (tensor<2x5x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-NEXT: onnx.Return [[RESULT]], [[TILED]]
// DISABLED-LABEL: func.func @fuse_tile_with_unrelated_use(
// DISABLED: [[TILED:%.+]] = "onnx.Tile"(%arg1,
// DISABLED: [[RESULT:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]])
// DISABLED: onnx.Return [[RESULT]], [[TILED]]

// -----

// The second consumer has a shortened non-axis output and must retain Tile.
func.func @fuse_tile_with_ineligible_gather_elements(
    %data: tensor<2x5x4xf32>, %other: tensor<3x5x4xf32>,
    %indices: tensor<1x3x1xi64>)
    -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>) {
  %repeats = onnx.Constant dense<[2, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %first = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  %second = "onnx.GatherElements"(%other, %tiled) {axis = 1 : si64} :
      (tensor<3x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %first, %second : tensor<2x3x4xf32>, tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @fuse_tile_with_ineligible_gather_elements(
// CHECK-DAG: [[TILED:%.+]] = "onnx.Tile"(%arg2,
// CHECK-DAG: [[FLAT:%.+]] = "onnx.Reshape"(%arg2, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x3x1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG: [[FIRST:%.+]] = "onnx.Gather"(%arg0, [[FLAT]]) {axis = 1 : si64} : (tensor<2x5x4xf32>, tensor<3xi64>) -> tensor<2x3x4xf32>
// CHECK-DAG: [[SECOND:%.+]] = "onnx.GatherElements"(%arg1, [[TILED]]) {axis = 1 : si64}
// CHECK: onnx.Return [[FIRST]], [[SECOND]]
// DISABLED-LABEL: func.func @fuse_tile_with_ineligible_gather_elements(
// DISABLED: [[TILED:%.+]] = "onnx.Tile"(%arg2,
// DISABLED-DAG: [[FIRST:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]])
// DISABLED-DAG: [[SECOND:%.+]] = "onnx.GatherElements"(%arg1, [[TILED]])
// DISABLED: onnx.Return [[FIRST]], [[SECOND]]

// -----

func.func @fuse_shared_expand(
    %data: tensor<1x4x1x3xf32>, %other: tensor<1x4x1x3xf32>,
    %indices: tensor<1x2x1x1xi64>)
    -> (tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>) {
  %shape = onnx.Constant dense<[1, 2, 1, 3]> : tensor<4xi64>
  %expanded = "onnx.Expand"(%indices, %shape) :
      (tensor<1x2x1x1xi64>, tensor<4xi64>) -> tensor<1x2x1x3xi64>
  %first = "onnx.GatherElements"(%data, %expanded) {axis = 1 : si64} :
      (tensor<1x4x1x3xf32>, tensor<1x2x1x3xi64>) -> tensor<1x2x1x3xf32>
  %second = "onnx.GatherElements"(%other, %expanded) {axis = 1 : si64} :
      (tensor<1x4x1x3xf32>, tensor<1x2x1x3xi64>) -> tensor<1x2x1x3xf32>
  onnx.Return %first, %second : tensor<1x2x1x3xf32>, tensor<1x2x1x3xf32>
}
// CHECK-LABEL: func.func @fuse_shared_expand(
// CHECK-NOT: "onnx.Expand"
// CHECK-NOT: "onnx.Tile"
// CHECK-NOT: "onnx.GatherElements"
// CHECK: [[FLAT:%.+]] = "onnx.Reshape"(%arg2, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x2x1x1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NEXT: [[FIRST:%.+]] = "onnx.Gather"(%arg0, [[FLAT]]) {axis = 1 : si64} : (tensor<1x4x1x3xf32>, tensor<2xi64>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT: [[OTHER_FLAT:%.+]] = "onnx.Reshape"(%arg2, %{{.+}}) {allowzero = 0 : si64} : (tensor<1x2x1x1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-NEXT: [[SECOND:%.+]] = "onnx.Gather"(%arg1, [[OTHER_FLAT]]) {axis = 1 : si64} : (tensor<1x4x1x3xf32>, tensor<2xi64>) -> tensor<1x2x1x3xf32>
// CHECK-NEXT: onnx.Return [[FIRST]], [[SECOND]]
// DISABLED-LABEL: func.func @fuse_shared_expand(
// DISABLED: [[TILED:%.+]] = "onnx.Tile"(%arg2,
// DISABLED-DAG: [[FIRST:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]])
// DISABLED-DAG: [[SECOND:%.+]] = "onnx.GatherElements"(%arg1, [[TILED]])
// DISABLED: onnx.Return [[FIRST]], [[SECOND]]

// -----

// Indices that vary along a non-axis dimension cannot become a flat Gather.
func.func @do_not_fuse_non_axis_dependent_indices(
    %data: tensor<2x5x4xf32>, %indices: tensor<2x3x1xi64>)
    -> tensor<2x3x4xf32> {
  %repeats = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<2x3x1xi64>, tensor<3xi64>) -> tensor<2x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<2x3x4xi64>) -> tensor<2x3x4xf32>
  onnx.Return %result : tensor<2x3x4xf32>
}
// CHECK-LABEL: func.func @do_not_fuse_non_axis_dependent_indices(
// CHECK: [[TILED:%.+]] = "onnx.Tile"
// CHECK-NEXT: [[RESULT:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]]) {axis = 1 : si64}
// CHECK-NEXT: onnx.Return [[RESULT]]

// -----

// Gather would retain the full first data dimension, unlike GatherElements.
func.func @do_not_fuse_partial_non_axis_dimension(
    %data: tensor<2x5x4xf32>, %indices: tensor<1x3x1xi64>)
    -> tensor<1x3x4xf32> {
  %repeats = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
  %tiled = "onnx.Tile"(%indices, %repeats) :
      (tensor<1x3x1xi64>, tensor<3xi64>) -> tensor<1x3x4xi64>
  %result = "onnx.GatherElements"(%data, %tiled) {axis = 1 : si64} :
      (tensor<2x5x4xf32>, tensor<1x3x4xi64>) -> tensor<1x3x4xf32>
  onnx.Return %result : tensor<1x3x4xf32>
}
// CHECK-LABEL: func.func @do_not_fuse_partial_non_axis_dimension(
// CHECK: [[TILED:%.+]] = "onnx.Tile"
// CHECK: [[RESULT:%.+]] = "onnx.GatherElements"(%arg0, [[TILED]]) {axis = 1 : si64}
// CHECK: onnx.Return [[RESULT]]
