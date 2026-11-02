// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --canonicalize-with-rn --enable-slice-canonicalization %s -split-input-file | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir-opt --canonicalize-with-rn %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// =============================================================================
// NormalizeSliceOperandsPattern
// =============================================================================

// -----

func.func @slice_defaults_materialized(%arg0: tensor<3x4x5xf32>) -> tensor<2x2x5xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %ends = onnx.Constant dense<[3, 5]> : tensor<2xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %none, %none) {onnx_node_name = "SliceDefaults"} : (tensor<3x4x5xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<2x2x5xf32>
  return %0 : tensor<2x2x5xf32>
// ENABLED-LABEL: func.func @slice_defaults_materialized
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<3x4x5xf32>) -> tensor<2x2x5xf32> {
// ENABLED-DAG:       [[STARTS:%.+]] = onnx.Constant dense<[1, 2, 0]> : tensor<3xi64>
// ENABLED-DAG:       [[ENDS:%.+]] = onnx.Constant dense<[3, 4, 5]> : tensor<3xi64>
// ENABLED-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// ENABLED-DAG:       [[STEPS:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]], [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) {onnx_node_name = "SliceDefaults"} : (tensor<3x4x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x2x5xf32>
// ENABLED:           return [[SLICE]] : tensor<2x2x5xf32>
}

// -----

func.func @slice_negative_values_normalized(%arg0: tensor<2x4x8xf32>) -> tensor<2x2x4xf32> {
  %starts = onnx.Constant dense<[-3, -7]> : tensor<2xi64>
  %ends = onnx.Constant dense<[-1, 100]> : tensor<2xi64>
  %axes = onnx.Constant dense<[-2, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x2x4xf32>
  return %0 : tensor<2x2x4xf32>
// ENABLED-LABEL: func.func @slice_negative_values_normalized
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<2x4x8xf32>) -> tensor<2x2x4xf32> {
// ENABLED-DAG:       [[STARTS:%.+]] = onnx.Constant dense<[0, 1, 1]> : tensor<3xi64>
// ENABLED-DAG:       [[ENDS:%.+]] = onnx.Constant dense<[2, 3, 8]> : tensor<3xi64>
// ENABLED-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// ENABLED-DAG:       [[STEPS:%.+]] = onnx.Constant dense<[1, 1, 2]> : tensor<3xi64>
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]], [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4x8xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x2x4xf32>
// ENABLED:           return [[SLICE]] : tensor<2x2x4xf32>
}

// -----

func.func @slice_negative_step_stable_sentinel(%arg0: tensor<2x5xf32>) -> tensor<2x3xf32> {
  %starts = onnx.Constant dense<-1> : tensor<1xi64>
  %ends = onnx.Constant dense<-6> : tensor<1xi64>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %steps = onnx.Constant dense<-2> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// ENABLED-LABEL: func.func @slice_negative_step_stable_sentinel
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<2x5xf32>) -> tensor<2x3xf32> {
// ENABLED-DAG:       [[STARTS:%.+]] = onnx.Constant dense<[0, 4]> : tensor<2xi64>
// ENABLED-DAG:       [[ENDS:%.+]] = onnx.Constant dense<[2, -6]> : tensor<2xi64>
// ENABLED-DAG:       [[AXES:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// ENABLED-DAG:       [[STEPS:%.+]] = onnx.Constant dense<[1, -2]> : tensor<2xi64>
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]], [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>
// ENABLED:           return [[SLICE]] : tensor<2x3xf32>
}

// =============================================================================
// FuseSliceSlicePattern
// =============================================================================

// -----

func.func @slice_slice_fusion(%arg0: tensor<10xf32>) -> tensor<3xf32> {
  %starts0 = onnx.Constant dense<2> : tensor<1xi64>
  %ends0 = onnx.Constant dense<8> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<6xf32>
  %starts1 = onnx.Constant dense<1> : tensor<1xi64>
  %ends1 = onnx.Constant dense<4> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts1, %ends1, %axes, %steps) : (tensor<6xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
// ENABLED-LABEL: func.func @slice_slice_fusion
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<10xf32>) -> tensor<3xf32> {
// ENABLED-DAG:       [[START:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// ENABLED-DAG:       [[END:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// ENABLED-DAG:       [[AXIS:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// ENABLED-DAG:       [[STEP:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]], [[START]], [[END]], [[AXIS]], [[STEP]]) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
// ENABLED:           return [[SLICE]] : tensor<3xf32>
}

// -----

func.func @fuse_slice_slice(%arg0: tensor<4x4xf32>) -> tensor<2x2xf32> {
  %starts0 = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends0 = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %inner = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps)
      : (tensor<4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xf32>
  %starts1 = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %ends1 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %outer = "onnx.Slice"(%inner, %starts1, %ends1, %axes, %steps)
      : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x2xf32>
  return %outer : tensor<2x2xf32>
// ENABLED-LABEL: func.func @fuse_slice_slice
// ENABLED-SAME:  (%[[X:.*]]: tensor<4x4xf32>)
// ENABLED-DAG:   %[[STARTS:.*]] = onnx.Constant dense<1> : tensor<2xi64>
// ENABLED-DAG:   %[[ENDS:.*]] = onnx.Constant dense<3> : tensor<2xi64>
// ENABLED-DAG:   %[[AXES:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// ENABLED:       %[[SLICE:.*]] = "onnx.Slice"(%[[X]], %[[STARTS]], %[[ENDS]], %[[AXES]]
// ENABLED:       return %[[SLICE]]
// ENABLED-NOT:   "onnx.Slice"{{.*}}"onnx.Slice"
}

// -----

func.func @slice_slice_strided_outer_not_fused(%arg0: tensor<10xf32>) -> tensor<3xf32> {
  %starts0 = onnx.Constant dense<2> : tensor<1xi64>
  %ends0 = onnx.Constant dense<9> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps0 = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps0) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<7xf32>
  %starts1 = onnx.Constant dense<0> : tensor<1xi64>
  %ends1 = onnx.Constant dense<6> : tensor<1xi64>
  %steps1 = onnx.Constant dense<2> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts1, %ends1, %axes, %steps1) : (tensor<7xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
// ENABLED-LABEL: func.func @slice_slice_strided_outer_not_fused
// ENABLED:         [[INNER:%.+]] = "onnx.Slice"(%arg0
// ENABLED:         [[OUTER:%.+]] = "onnx.Slice"([[INNER]]
// ENABLED:         return [[OUTER]] : tensor<3xf32>
}

// -----

func.func @fuse_slice_slice_inner_multi_use(%arg0: tensor<4x4xf32>) -> (tensor<2x4xf32>, tensor<2x2xf32>) {
  %starts0 = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends0 = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %inner = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps)
      : (tensor<4x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xf32>
  %starts1 = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %ends1 = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %outer = "onnx.Slice"(%inner, %starts1, %ends1, %axes, %steps)
      : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x2xf32>
  return %inner, %outer : tensor<2x4xf32>, tensor<2x2xf32>
// ENABLED-LABEL: func.func @fuse_slice_slice_inner_multi_use
// ENABLED-SAME:  (%[[ARG0:.*]]: tensor<4x4xf32>)
// ENABLED:       %[[INNER:.*]] = "onnx.Slice"(%[[ARG0]]
// ENABLED:       %[[FUSED:.*]] = "onnx.Slice"(%[[ARG0]]
// ENABLED:       return %[[INNER]], %[[FUSED]]
}

// -----

func.func @slice_slice_flag_off(%arg0: tensor<10xf32>) -> tensor<3xf32> {
  %starts0 = onnx.Constant dense<2> : tensor<1xi64>
  %ends0 = onnx.Constant dense<8> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts0, %ends0, %axes, %steps) : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<6xf32>
  %starts1 = onnx.Constant dense<1> : tensor<1xi64>
  %ends1 = onnx.Constant dense<4> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts1, %ends1, %axes, %steps) : (tensor<6xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
// DISABLED-LABEL: func.func @slice_slice_flag_off
// DISABLED:         [[INNER:%.+]] = "onnx.Slice"(%arg0
// DISABLED:         [[OUTER:%.+]] = "onnx.Slice"([[INNER]]
// DISABLED:         return [[OUTER]] : tensor<3xf32>
}

// =============================================================================
// SliceTileIdentityPattern
// =============================================================================

// -----

func.func @slice_tile_identity_folds(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %repeats = onnx.Constant dense<3> : tensor<1xi64>
  %0 = "onnx.Tile"(%arg0, %repeats) : (tensor<4xf32>, tensor<1xi64>) -> tensor<12xf32>
  %starts = onnx.Constant dense<4> : tensor<1xi64>
  %ends = onnx.Constant dense<8> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
// ENABLED-LABEL: func.func @slice_tile_identity_folds
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<4xf32>) -> tensor<4xf32> {
// ENABLED-NOT:       "onnx.Tile"
// ENABLED-NOT:       "onnx.Slice"
// ENABLED:           return [[ARG0]] : tensor<4xf32>
}

// -----

func.func @slice_tile_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %repeats = onnx.Constant dense<[3, 1]> : tensor<2xi64>
  %tiled = "onnx.Tile"(%arg0, %repeats) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<6x3xf32>
  %starts = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[4, 3]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%tiled, %starts, %ends, %axes, %steps)
      : (tensor<6x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>
  return %sliced : tensor<2x3xf32>
// ENABLED-LABEL: func.func @slice_tile_identity
// ENABLED-SAME:  (%[[X:.*]]: tensor<2x3xf32>)
// ENABLED-NOT:   "onnx.Tile"
// ENABLED-NOT:   "onnx.Slice"
// ENABLED:       return %[[X]]
}

// -----

func.func @slice_tile_partial_reduces(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %repeats = onnx.Constant dense<[3, 1]> : tensor<2xi64>
  %tiled = "onnx.Tile"(%arg0, %repeats) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<6x3xf32>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[3, 3]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%tiled, %starts, %ends, %axes, %steps)
      : (tensor<6x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>
  return %sliced : tensor<2x3xf32>
// The slice spans 1.5 copies, so the tile shrinks from 3 to 2 copies on axis 0.
// ENABLED-LABEL: func.func @slice_tile_partial_reduces
// ENABLED:         [[TILE:%.+]] = "onnx.Tile"(%arg0, {{.*}}) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<4x3xf32>
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[TILE]]
// ENABLED:         return [[SLICE]] : tensor<2x3xf32>
}

// -----

func.func @slice_tile_multi_use_not_reduced(%arg0: tensor<4xf32>) -> (tensor<12xf32>, tensor<4xf32>) {
  %repeats = onnx.Constant dense<3> : tensor<1xi64>
  %0 = "onnx.Tile"(%arg0, %repeats) : (tensor<4xf32>, tensor<1xi64>) -> tensor<12xf32>
  %starts = onnx.Constant dense<4> : tensor<1xi64>
  %ends = onnx.Constant dense<8> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<12xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %0, %1 : tensor<12xf32>, tensor<4xf32>
// ENABLED-LABEL: func.func @slice_tile_multi_use_not_reduced
// ENABLED:         [[TILE:%.+]] = "onnx.Tile"(%arg0
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[TILE]]
// ENABLED:         return [[TILE]], [[SLICE]]
}

// -----

func.func @slice_tile_repeats_not_reduced(%arg0: tensor<4xf32>) -> tensor<6xf32> {
  %repeats = onnx.Constant dense<2> : tensor<1xi64>
  %0 = "onnx.Tile"(%arg0, %repeats) : (tensor<4xf32>, tensor<1xi64>) -> tensor<8xf32>
  %starts = onnx.Constant dense<1> : tensor<1xi64>
  %ends = onnx.Constant dense<7> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<6xf32>
  return %1 : tensor<6xf32>
// ENABLED-LABEL: func.func @slice_tile_repeats_not_reduced
// ENABLED:         [[TILE:%.+]] = "onnx.Tile"(%arg0
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[TILE]]
// ENABLED:         return [[SLICE]] : tensor<6xf32>
}

// =============================================================================
// SlicePadDropPattern
// =============================================================================

// -----

func.func @slice_pad_drops_pad(%arg0: tensor<6xf32>) -> tensor<4xf32> {
  %pads = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, none, none) -> tensor<11xf32>
  %starts = onnx.Constant dense<3> : tensor<1xi64>
  %ends = onnx.Constant dense<7> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<11xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
// ENABLED-LABEL: func.func @slice_pad_drops_pad
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<6xf32>) -> tensor<4xf32> {
// ENABLED-DAG:       [[START:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// ENABLED-DAG:       [[END:%.+]] = onnx.Constant dense<5> : tensor<1xi64>
// ENABLED-DAG:       [[AXIS:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]], [[START]], [[END]], [[AXIS]]
// ENABLED:           return [[SLICE]] : tensor<4xf32>
}

// -----

func.func @slice_pad_drop(%arg0: tensor<2x3xf32>) -> tensor<1x2xf32> {
  %pads = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %padded = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"}
      : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<4x5xf32>
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%padded, %starts, %ends, %axes, %steps)
      : (tensor<4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  return %sliced : tensor<1x2xf32>
// ENABLED-LABEL: func.func @slice_pad_drop
// ENABLED-SAME:  (%[[X:.*]]: tensor<2x3xf32>)
// ENABLED-DAG:   %[[STARTS:.*]] = onnx.Constant dense<0> : tensor<2xi64>
// ENABLED-DAG:   %[[ENDS:.*]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// ENABLED-DAG:   %[[AXES:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// ENABLED:       %[[SLICE:.*]] = "onnx.Slice"(%[[X]], %[[STARTS]], %[[ENDS]], %[[AXES]]
// ENABLED:       return %[[SLICE]]
// ENABLED-NOT:   "onnx.Pad"
}

// -----

func.func @slice_pad_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %pads = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %padded = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"}
      : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<4x5xf32>
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%padded, %starts, %ends, %axes, %steps)
      : (tensor<4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>
  return %sliced : tensor<2x3xf32>
// ENABLED-LABEL: func.func @slice_pad_identity
// ENABLED-SAME:  (%[[X:.*]]: tensor<2x3xf32>)
// ENABLED-NOT:   "onnx.Pad"
// ENABLED-NOT:   "onnx.Slice"
// ENABLED:       return %[[X]]
}

// -----

func.func @slice_pad_reads_padding_reduces(%arg0: tensor<6xf32>) -> tensor<4xf32> {
  %pads = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, none, none) -> tensor<11xf32>
  %starts = onnx.Constant dense<1> : tensor<1xi64>
  %ends = onnx.Constant dense<5> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<11xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
// The window reads 1 leading pad + first 3 input elements, so the pad shrinks to
// low=1/high=0 and the input is sliced down: slice(pad(x)) -> pad(slice(x)).
// ENABLED-LABEL: func.func @slice_pad_reads_padding_reduces
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<6xf32>) -> tensor<4xf32> {
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]]
// ENABLED:           [[PAD:%.+]] = "onnx.Pad"([[SLICE]]
// ENABLED:           return [[PAD]] : tensor<4xf32>
}

// -----

func.func @slice_pad_reduces_pad_only(%arg0: tensor<2x3xf32>) -> tensor<3x3xf32> {
  %pads = onnx.Constant dense<[1, 1, 1, 1]> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %padded = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"}
      : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<4x5xf32>
  %starts = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%padded, %starts, %ends, %axes, %steps)
      : (tensor<4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x3xf32>
  return %sliced : tensor<3x3xf32>
// The window covers the whole input so no input slice is needed; only the pad
// amount shrinks (drops the trailing/leading padding the window no longer reads).
// ENABLED-LABEL: func.func @slice_pad_reduces_pad_only
// ENABLED-SAME:  (%[[X:.*]]: tensor<2x3xf32>)
// ENABLED-NOT:   "onnx.Slice"
// ENABLED:       %[[PAD:.*]] = "onnx.Pad"(%[[X]]
// ENABLED:       return %[[PAD]] : tensor<3x3xf32>
}

// -----

func.func @slice_pad_reads_padding_both(%arg0: tensor<2x3xf32>) -> tensor<4x3xf32> {
  %pads = onnx.Constant dense<[2, 2, 2, 2]> : tensor<4xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %padded = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"}
      : (tensor<2x3xf32>, tensor<4xi64>, none, none) -> tensor<6x7xf32>
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[5, 4]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%padded, %starts, %ends, %axes, %steps)
      : (tensor<6x7xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x3xf32>
  return %sliced : tensor<4x3xf32>
// Reads padding on both axes and part of the input, so both a reduced input
// slice and a reduced pad remain: slice(pad(x)) -> pad(slice(x)).
// ENABLED-LABEL: func.func @slice_pad_reads_padding_both
// ENABLED-SAME:  (%[[X:.*]]: tensor<2x3xf32>)
// ENABLED:       %[[SLICE:.*]] = "onnx.Slice"(%[[X]]
// ENABLED:       %[[PAD:.*]] = "onnx.Pad"(%[[SLICE]]
// ENABLED:       return %[[PAD]] : tensor<4x3xf32>
}

// -----

func.func @slice_pad_reflect_drops_pad(%arg0: tensor<6xf32>) -> tensor<4xf32> {
  %pads = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "reflect"} : (tensor<6xf32>, tensor<2xi64>, none, none) -> tensor<11xf32>
  %starts = onnx.Constant dense<3> : tensor<1xi64>
  %ends = onnx.Constant dense<7> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<11xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
// Reads only the unpadded input, so dropping the pad is valid for any mode.
// ENABLED-LABEL: func.func @slice_pad_reflect_drops_pad
// ENABLED-SAME:    ([[ARG0:%.+]]: tensor<6xf32>) -> tensor<4xf32> {
// ENABLED-NOT:       "onnx.Pad"
// ENABLED:           [[SLICE:%.+]] = "onnx.Slice"([[ARG0]]
// ENABLED:           return [[SLICE]] : tensor<4xf32>
}

// -----

func.func @slice_pad_reflect_reads_padding_kept(%arg0: tensor<6xf32>) -> tensor<4xf32> {
  %pads = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "reflect"} : (tensor<6xf32>, tensor<2xi64>, none, none) -> tensor<11xf32>
  %starts = onnx.Constant dense<1> : tensor<1xi64>
  %ends = onnx.Constant dense<5> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<11xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
// Re-padding a reflect pad would change values, so this must NOT be rewritten.
// ENABLED-LABEL: func.func @slice_pad_reflect_reads_padding_kept
// ENABLED:         [[PAD:%.+]] = "onnx.Pad"(%arg0
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[PAD]]
// ENABLED:         return [[SLICE]] : tensor<4xf32>
}

// -----

func.func @slice_pad_multi_use_not_reduced(%arg0: tensor<6xf32>) -> (tensor<11xf32>, tensor<4xf32>) {
  %pads = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %pads, %none, %none) {mode = "constant"} : (tensor<6xf32>, tensor<2xi64>, none, none) -> tensor<11xf32>
  %starts = onnx.Constant dense<3> : tensor<1xi64>
  %ends = onnx.Constant dense<7> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<11xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>
  return %0, %1 : tensor<11xf32>, tensor<4xf32>
// ENABLED-LABEL: func.func @slice_pad_multi_use_not_reduced
// ENABLED:         [[PAD:%.+]] = "onnx.Pad"(%arg0
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[PAD]]
// ENABLED:         return [[PAD]], [[SLICE]]
}

// =============================================================================
// SliceConcatExactCancelPattern
// =============================================================================

// -----

func.func @slice_concat_exact_cancel(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<4xf32>) -> tensor<3xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<9xf32>
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends = onnx.Constant dense<5> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
// ENABLED-LABEL: func.func @slice_concat_exact_cancel
// ENABLED-SAME:    (%arg0: tensor<2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<4xf32>) -> tensor<3xf32> {
// ENABLED:           return %arg1 : tensor<3xf32>
}

// -----

func.func @slice_concat_cancel(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<4x3xf32> {
  %concat = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64}
      : (tensor<2x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<7x3xf32>
  %starts = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[6, 3]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%concat, %starts, %ends, %axes, %steps)
      : (tensor<7x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<4x3xf32>
  return %sliced : tensor<4x3xf32>
// ENABLED-LABEL: func.func @slice_concat_cancel
// ENABLED-SAME:  ({{.*}}, %[[B:.*]]: tensor<4x3xf32>, {{.*}})
// ENABLED-NOT:   "onnx.Concat"
// ENABLED-NOT:   "onnx.Slice"
// ENABLED:       return %[[B]]
}

// -----

func.func @slice_concat_partial_not_cancelled(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
  %starts = onnx.Constant dense<1> : tensor<1xi64>
  %ends = onnx.Constant dense<4> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
// ENABLED-LABEL: func.func @slice_concat_partial_not_cancelled
// ENABLED:         [[CONCAT:%.+]] = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64}
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[CONCAT]]
// ENABLED:         return [[SLICE]] : tensor<3xf32>
}

// -----

func.func @slice_concat_partial(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<1x3xf32>) -> tensor<5x3xf32> {
  %concat = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64}
      : (tensor<2x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<7x3xf32>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[6, 3]> : tensor<2xi64>
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %sliced = "onnx.Slice"(%concat, %starts, %ends, %axes, %steps)
      : (tensor<7x3xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x3xf32>
  return %sliced : tensor<5x3xf32>
// The window covers arg0 and arg1 but not arg2, so the third operand is dropped.
// ENABLED-LABEL: func.func @slice_concat_partial
// ENABLED:         [[CONCAT:%.+]] = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} : (tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<6x3xf32>
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[CONCAT]]
// ENABLED:         return [[SLICE]] : tensor<5x3xf32>
}

// -----

func.func @slice_concat_multi_use_not_reduced(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<4xf32>) -> (tensor<9xf32>, tensor<5xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<9xf32>
  %starts = onnx.Constant dense<1> : tensor<1xi64>
  %ends = onnx.Constant dense<6> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xf32>
  return %0, %1 : tensor<9xf32>, tensor<5xf32>
// Reducing would keep 2 operands (arg0, arg1) but the concat has another user,
// so we must not build a second concat.
// ENABLED-LABEL: func.func @slice_concat_multi_use_not_reduced
// ENABLED:         [[CONCAT:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2)
// ENABLED:         [[SLICE:%.+]] = "onnx.Slice"([[CONCAT]]
// ENABLED:         return [[CONCAT]], [[SLICE]]
}

// -----

func.func @slice_concat_single_operand_multi_use(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<4xf32>) -> (tensor<9xf32>, tensor<3xf32>) {
  %0 = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<2xf32>, tensor<3xf32>, tensor<4xf32>) -> tensor<9xf32>
  %starts = onnx.Constant dense<2> : tensor<1xi64>
  %ends = onnx.Constant dense<5> : tensor<1xi64>
  %axes = onnx.Constant dense<0> : tensor<1xi64>
  %steps = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<9xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
  return %0, %1 : tensor<9xf32>, tensor<3xf32>
// Keeping a single operand introduces no new concat, so it fires even though the
// original concat is multi-use: slice collapses to arg1.
// ENABLED-LABEL: func.func @slice_concat_single_operand_multi_use
// ENABLED:         [[CONCAT:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2)
// ENABLED:         return [[CONCAT]], %arg1 : tensor<9xf32>, tensor<3xf32>
}
