// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --enable-reshape-canonicalization=false --onnx-hybrid-transform="shape-inference=false constant-propagation=false decomposition=false" %s -split-input-file | FileCheck %s

// -----

func.func @positive_axis_concat(%arg0: tensor<2x3xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x7xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  return %0 : tensor<2x7xf32>
// CHECK-LABEL: func.func @positive_axis_concat
// CHECK: "onnx.Concat"{{.*}} {axis = 1 : si64}
}

// -----

func.func @positive_axis_softmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axis_softmax
// CHECK: "onnx.Softmax"{{.*}} {axis = 1 : si64}
}

func.func @positive_axis_onehot(%indices: tensor<2x3xi64>, %depth: tensor<i64>, %values: tensor<2xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.OneHot"(%indices, %depth, %values) {axis = -1 : si64} : (tensor<2x3xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_onehot
// CHECK: "onnx.OneHot"{{.*}} {axis = 2 : si64}
}

// -----

func.func @positive_axis_cumsum(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %axis = onnx.Constant dense<-1> : tensor<i64>
  %0 = "onnx.CumSum"(%arg0, %axis) : (tensor<2x3x4xf32>, tensor<i64>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_cumsum
// CHECK: onnx.Constant dense<2> : tensor<i64>
// CHECK: "onnx.CumSum"
}

// -----

func.func @positive_axes_reduce_operand(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
  %axes = onnx.Constant dense<[-1, -2]> : tensor<2xi64>
  %0 = "onnx.ReduceMean"(%arg0, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
  return %0 : tensor<2x1x1xf32>
// CHECK-LABEL: func.func @positive_axes_reduce_operand
// CHECK: onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK: "onnx.ReduceMean"
}

// -----

func.func @positive_axes_reduce_attr(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [-1, -2], keepdims = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<2x1x1xf32>
  return %0 : tensor<2x1x1xf32>
// CHECK-LABEL: func.func @positive_axes_reduce_attr
// CHECK: onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK: "onnx.ReduceMean"{{.*}} {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
// CHECK-NOT: "onnx.ReduceMeanV13"
}

// -----

func.func @positive_axis_guards(%arg0: tensor<*xf32>, %arg1: tensor<2x3x4xf32>, %dynamic_axes: tensor<1xi64>) -> (tensor<*xf32>, tensor<1x1x1xf32>, tensor<2x3x4xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "onnx.ReduceMean"(%arg1, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  %2 = "onnx.ReduceMean"(%arg1, %dynamic_axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x3x4xf32>
  return %0, %1, %2 : tensor<*xf32>, tensor<1x1x1xf32>, tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_guards
// CHECK: "onnx.Softmax"{{.*}} {axis = -1 : si64}
// CHECK: "onnx.ReduceMean"{{.*}}keepdims = 1 : si64
// CHECK: "onnx.ReduceMean"{{.*}}keepdims = 1 : si64
}

// -----

func.func @positive_axis_resize_axes_attr(%arg0: tensor<2x3xf32>) -> tensor<2x6xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = onnx.Constant dense<[2, 6]> : tensor<2xi64>
  %0 = "onnx.Resize"(%arg0, %none, %none, %sizes) {axes = [-1], mode = "nearest"} : (tensor<2x3xf32>, none, none, tensor<2xi64>) -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
// CHECK-LABEL: func.func @positive_axis_resize_axes_attr
// CHECK: "onnx.Resize"{{.*}}axes = [1]
}

// -----

func.func @positive_axis_argmax(%arg0: tensor<2x3x4xf32>) -> tensor<*xi64> {
  %0 = "onnx.ArgMax"(%arg0) {axis = -1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>
// CHECK-LABEL: func.func @positive_axis_argmax
// CHECK: "onnx.ArgMax"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_argmin(%arg0: tensor<2x3x4xf32>) -> tensor<*xi64> {
  %0 = "onnx.ArgMin"(%arg0) {axis = -1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>
// CHECK-LABEL: func.func @positive_axis_argmin
// CHECK: "onnx.ArgMin"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_compress(%arg0: tensor<2x3xf32>, %cond: tensor<3xi1>) -> tensor<2x?xf32> {
  %0 = "onnx.Compress"(%arg0, %cond) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<3xi1>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
// CHECK-LABEL: func.func @positive_axis_compress
// CHECK: "onnx.Compress"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_concat_from_sequence(%arg0: !onnx.Seq<tensor<2x3xf32>>) -> tensor<*xf32> {
  %0 = "onnx.ConcatFromSequence"(%arg0) {axis = -1 : si64, new_axis = 0 : si64} : (!onnx.Seq<tensor<2x3xf32>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL: func.func @positive_axis_concat_from_sequence
// CHECK: "onnx.ConcatFromSequence"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_dequantize_linear(%arg0: tensor<2x3xui8>, %scale: tensor<f32>, %zero: tensor<ui8>) -> tensor<2x3xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %scale, %zero) {axis = -1 : si64, block_size = 0 : si64} : (tensor<2x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axis_dequantize_linear
// CHECK: "onnx.DequantizeLinear"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_flatten(%arg0: tensor<2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Flatten"(%arg0) {axis = -1 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL: func.func @positive_axis_flatten
// CHECK: "onnx.Flatten"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_gather(%arg0: tensor<2x3xf32>, %indices: tensor<2xi64>) -> tensor<2x2xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
// CHECK-LABEL: func.func @positive_axis_gather
// CHECK: "onnx.Gather"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_gather_elements(%arg0: tensor<2x3xf32>, %indices: tensor<2x3xi64>) -> tensor<2x3xf32> {
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x3xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axis_gather_elements
// CHECK: "onnx.GatherElements"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_hardmax(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = -1 : si64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_hardmax
// CHECK: "onnx.Hardmax"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_layer_normalization(%arg0: tensor<2x3x4xf32>, %scale: tensor<4xf32>, %bias: tensor<4xf32>) -> tensor<2x3x4xf32> {
  %0, %1, %2 = "onnx.LayerNormalization"(%arg0, %scale, %bias) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<2x3x4xf32>, none, none)
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_layer_normalization
// CHECK: "onnx.LayerNormalization"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_logsoftmax(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.LogSoftmax"(%arg0) {axis = -1 : si64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_logsoftmax
// CHECK: "onnx.LogSoftmax"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_lp_normalization(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.LpNormalization"(%arg0) {axis = -1 : si64, p = 2 : si64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
// CHECK-LABEL: func.func @positive_axis_lp_normalization
// CHECK: "onnx.LpNormalization"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_quantize_linear(%arg0: tensor<2x3xf32>, %scale: tensor<f32>, %zero: tensor<ui8>) -> tensor<2x3xui8> {
  %0 = "onnx.QuantizeLinear"(%arg0, %scale, %zero) {axis = -1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x3xui8>
  return %0 : tensor<2x3xui8>
// CHECK-LABEL: func.func @positive_axis_quantize_linear
// CHECK: "onnx.QuantizeLinear"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_scatter(%data: tensor<2x3xf32>, %indices: tensor<2x3xi64>, %updates: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Scatter"(%data, %indices, %updates) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x3xi64>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axis_scatter
// CHECK: "onnx.Scatter"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_scatter_elements(%data: tensor<2x3xf32>, %indices: tensor<2x3xi64>, %updates: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.ScatterElements"(%data, %indices, %updates) {axis = -1 : si64} : (tensor<2x3xf32>, tensor<2x3xi64>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axis_scatter_elements
// CHECK: "onnx.ScatterElements"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_split(%arg0: tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:2 = "onnx.Split"(%arg0, %none) {axis = -1 : si64} : (tensor<2x4xf32>, none) -> (tensor<2x2xf32>, tensor<2x2xf32>)
  return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xf32>
// CHECK-LABEL: func.func @positive_axis_split
// CHECK: "onnx.Split"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_split_v11(%arg0: tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %0:2 = "onnx.SplitV11"(%arg0) {axis = -1 : si64} : (tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)
  return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xf32>
// CHECK-LABEL: func.func @positive_axis_split_v11
// CHECK: "onnx.SplitV11"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_split_v13(%arg0: tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:2 = "onnx.SplitV13"(%arg0, %none) {axis = -1 : si64} : (tensor<2x4xf32>, none) -> (tensor<2x2xf32>, tensor<2x2xf32>)
  return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xf32>
// CHECK-LABEL: func.func @positive_axis_split_v13
// CHECK: "onnx.SplitV13"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_split_to_sequence(%arg0: tensor<2x4xf32>) -> !onnx.Seq<tensor<2x1xf32>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.SplitToSequence"(%arg0, %none) {axis = -1 : si64, keepdims = 1 : si64} : (tensor<2x4xf32>, none) -> !onnx.Seq<tensor<2x1xf32>>
  return %0 : !onnx.Seq<tensor<2x1xf32>>
// CHECK-LABEL: func.func @positive_axis_split_to_sequence
// CHECK: "onnx.SplitToSequence"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axis_topk(%arg0: tensor<2x3x4xf32>, %k: tensor<i64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %values, %indices = "onnx.TopK"(%arg0, %k) {axis = -1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<2x3x4xf32>, tensor<i64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %values, %indices : tensor<*xf32>, tensor<*xi64>
// CHECK-LABEL: func.func @positive_axis_topk
// CHECK: "onnx.TopK"{{.*}}axis = 2 : si64
}

// -----

func.func @positive_axis_unique(%arg0: tensor<2x3xi64>) -> tensor<*xi64> {
  %y, %indices, %inverse_indices, %counts = "onnx.Unique"(%arg0) {axis = -1 : si64, sorted = 1 : si64} : (tensor<2x3xi64>) -> (tensor<*xi64>, none, none, none)
  return %y : tensor<*xi64>
// CHECK-LABEL: func.func @positive_axis_unique
// CHECK: "onnx.Unique"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axes_squeeze_v11_attr(%arg0: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) {axes = [-2]} : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axes_squeeze_v11_attr
// CHECK: "onnx.SqueezeV11"{{.*}}axes = [1]
}

// -----

func.func @positive_axes_unsqueeze_v11_attr(%arg0: tensor<2x3xf32>) -> tensor<2x3x1xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes = [-1]} : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
  return %0 : tensor<2x3x1xf32>
// CHECK-LABEL: func.func @positive_axes_unsqueeze_v11_attr
// CHECK: "onnx.UnsqueezeV11"{{.*}}axes = [2]
}

// -----

func.func @positive_axis_dft_operand(%arg0: tensor<?x?x?xf32>, %length: tensor<?xi64>) -> tensor<*xf32> {
  %axis = onnx.Constant dense<-1> : tensor<i64>
  %0 = "onnx.DFT"(%arg0, %length, %axis) {inverse = 0 : si64, onesided = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>, tensor<i64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL: func.func @positive_axis_dft_operand
// CHECK: onnx.Constant dense<1> : tensor<i64>
// CHECK: "onnx.DFT"
}

// -----

func.func @positive_axis_dft_v17_attr(%arg0: tensor<?x?x?xf32>, %length: tensor<?xi64>) -> tensor<*xf32> {
  %0 = "onnx.DFTV17"(%arg0, %length) {axis = -1 : si64, inverse = 0 : si64, onesided = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL: func.func @positive_axis_dft_v17_attr
// CHECK: "onnx.DFTV17"{{.*}}axis = 1 : si64
}

// -----

func.func @positive_axes_pad_operand(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %pads = onnx.Constant dense<[0, 0]> : tensor<2xi64>
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axes_pad_operand
// CHECK: onnx.Constant dense<1> : tensor<1xi64>
// CHECK: "onnx.Pad"
}

// -----

func.func @positive_axes_pad_v18_operand(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %pads = onnx.Constant dense<[0, 0]> : tensor<2xi64>
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %0 = "onnx.PadV18"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axes_pad_v18_operand
// CHECK: onnx.Constant dense<1> : tensor<1xi64>
// CHECK: "onnx.PadV18"
}

// -----

func.func @positive_axes_slice_operand(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x2xf32> {
  %starts = onnx.Constant dense<[0]> : tensor<1xi64>
  %ends = onnx.Constant dense<[2]> : tensor<1xi64>
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3x2xf32>
  return %0 : tensor<2x3x2xf32>
// CHECK-LABEL: func.func @positive_axes_slice_operand
// CHECK: onnx.Constant dense<2> : tensor<1xi64>
// CHECK: "onnx.Slice"
}

// -----

func.func @positive_axes_squeeze_operand(%arg0: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
  %axes = onnx.Constant dense<[-2]> : tensor<1xi64>
  %0 = "onnx.Squeeze"(%arg0, %axes) : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL: func.func @positive_axes_squeeze_operand
// CHECK: onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK: "onnx.Reshape"
}

// -----

func.func @positive_axes_unsqueeze_operand(%arg0: tensor<2x3xf32>) -> tensor<2x3x1xf32> {
  %axes = onnx.Constant dense<[-1]> : tensor<1xi64>
  %0 = "onnx.Unsqueeze"(%arg0, %axes) : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x3x1xf32>
  return %0 : tensor<2x3x1xf32>
// CHECK-LABEL: func.func @positive_axes_unsqueeze_operand
// CHECK: onnx.Constant dense<[2, 3, 1]> : tensor<3xi64>
// CHECK: "onnx.Reshape"
}
