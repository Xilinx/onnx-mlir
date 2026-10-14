// Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// -----

// Layernorm with bias (not recognized as need multiple passes).

func.func @layernorm_with_spurious_adds(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %x = "onnx.Add"(%input, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %output = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %output : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_with_spurious_adds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm without bias
func.func @layernorm_without_bias(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_first_reduce_unsuitable_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-2], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_first_reduce_unsuitable_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-2> : tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_3_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_second_reduce_unsuitable_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-2], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_second_reduce_unsuitable_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[VAR_Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %axis = onnx.Constant dense<-1> : tensor<1xi64>
  %mean = "onnx.ReduceMean"(%x, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_dynamic_axis(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>, %axis: tensor<?xi64>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMean"(%x, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_dynamic_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>, [[PARAM_3_:%.+]]: tensor<?xi64>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[PARAM_3_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[PARAM_3_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<?xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_0_]]) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sqrt"([[VAR_5_]]) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Div"([[VAR_2_]], [[VAR_6_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Mul"([[VAR_7_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_8_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_first_reduce_unsuitable_axis_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %axis1 = onnx.Constant dense<-2> : tensor<1xi64>
  %axis2 = onnx.Constant dense<-1> : tensor<1xi64>
  %mean = "onnx.ReduceMean"(%x, %axis1) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis2) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_first_reduce_unsuitable_axis_v18
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-2> : tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_3_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_second_reduce_unsuitable_axis_v18(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %axis1 = onnx.Constant dense<-1> : tensor<1xi64>
  %axis2 = onnx.Constant dense<-2> : tensor<1xi64>
  %mean = "onnx.ReduceMean"(%x, %axis1) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %axis2) {keepdims = 1 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_second_reduce_unsuitable_axis_v18
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[VAR_Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_noop(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %mean = "onnx.ReduceMean"(%x, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x768xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x768xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x768xf32>, tensor<f32>) -> tensor<1x384x768xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_1_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_2_]], [[VAR_0_]]) : (tensor<1x384x768xf32>, tensor<f32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sqrt"([[VAR_3_]]) : (tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Div"([[VAR_1_]], [[VAR_4_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_5_]], [[PARAM_1_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

func.func @layernorm_without_bias_v18_reduce_all(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %mean = "onnx.ReduceMean"(%x, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMean"(%dd, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0: si64} : (tensor<1x384x768xf32>, none) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_without_bias_v18_reduce_all
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 0 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Layernorm, add/mul switched

func.func @layernorm_with_bias_switched(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_with_bias_switched
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}


// -----

// Not a Layernorm as top sub has inputs switched
func.func @not_a_layer_norm(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%mean, %x) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @not_a_layer_norm
// CHECK-NOT:       "onnx.LayerNormalization"
// CHECK:         }
}

// -----
// Check alternative layer norm with reciprocal instead of div
func.func @layer_norm_with_reciprocal(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %x = "onnx.Add"(%input, %input)  : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %InvStdDev = "onnx.Reciprocal"(%StdDev) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Mul"(%d, %InvStdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %res = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %res : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layer_norm_with_reciprocal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Check alternative layer norm with reciprocal instead of div
func.func @layer_norm_with_div_by_one(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %one = onnx.Constant dense<1.0> : tensor<f32>
  %x = "onnx.Add"(%input, %input)  : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %InvStdDev = "onnx.Div"(%one, %StdDev) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Mul"(%d, %InvStdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %res = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %res : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layer_norm_with_div_by_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[Y_]], [[PARAM_2_]]) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// Check alternative layer norm with reciprocal instead of div, fail because it is 2 / x instead of 1 / x
func.func @not_a_layer_norm_with_div_by_two(%input: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %one = onnx.Constant dense<2.0> : tensor<f32>
  %x = "onnx.Add"(%input, %input)  : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %InvStdDev = "onnx.Div"(%one, %StdDev) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Mul"(%d, %InvStdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  %res = "onnx.Add"(%Y, %bias) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32>
  return %res : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @not_a_layer_norm_with_div_by_two
// CHECK-NOT:       "onnx.LayerNormalization"
// CHECK:         }
}

// -----

func.func @layernorm_with_double_mul(%arg0: tensor<1x1370x384xf32>, %arg1: tensor<1x1370x384xf32>, %arg2: tensor<384xf32>, %arg3: tensor<384xf32>) -> (tensor<1x1370x384xf32>, tensor<1x1370x384xf32>) {
  %0 = onnx.Constant dense<3.0> : tensor<f32>
  %1 = "onnx.Add"(%arg0, %arg1) : (tensor<1x1370x384xf32>, tensor<1x1370x384xf32>) -> tensor<1x1370x384xf32>
  %2 = "onnx.ReduceMeanV13"(%1) {axes = [-1], keepdims = 1 : si64} : (tensor<1x1370x384xf32>) -> tensor<1x1370x1xf32>
  %3 = "onnx.Sub"(%1, %2) : (tensor<1x1370x384xf32>, tensor<1x1370x1xf32>) -> tensor<1x1370x384xf32>
  %4 = "onnx.Mul"(%3, %3) : (tensor<1x1370x384xf32>, tensor<1x1370x384xf32>) -> tensor<1x1370x384xf32>
  %5 = "onnx.ReduceMeanV13"(%4) {axes = [-1], keepdims = 1 : si64} : (tensor<1x1370x384xf32>) -> tensor<1x1370x1xf32>
  %6 = "onnx.Add"(%5, %0) : (tensor<1x1370x1xf32>, tensor<f32>) -> tensor<1x1370x1xf32>
  %7 = "onnx.Sqrt"(%6) : (tensor<1x1370x1xf32>) -> tensor<1x1370x1xf32>
  %8 = "onnx.Div"(%3, %7) : (tensor<1x1370x384xf32>, tensor<1x1370x1xf32>) -> tensor<1x1370x384xf32>
  %9 = "onnx.Mul"(%8, %arg2) : (tensor<1x1370x384xf32>, tensor<384xf32>) -> tensor<1x1370x384xf32>
  %10 = "onnx.Mul"(%8, %arg3) : (tensor<1x1370x384xf32>, tensor<384xf32>) -> tensor<1x1370x384xf32>
  return %9, %10: tensor<1x1370x384xf32>, tensor<1x1370x384xf32>
// CHECK-LABEL:  func.func @layernorm_with_double_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1370x384xf32>, [[PARAM_1_:%.+]]: tensor<1x1370x384xf32>, [[PARAM_2_:%.+]]: tensor<384xf32>, [[PARAM_3_:%.+]]: tensor<384xf32>) -> (tensor<1x1370x384xf32>, tensor<1x1370x384xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<384xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1x1370x384xf32>, tensor<1x1370x384xf32>) -> tensor<1x1370x384xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_2_]], [[VAR_0_]], [[VAR_1_]]) {axis = 2 : si64, epsilon = 3.000000e+00 : f32, stash_type = 1 : si64} : (tensor<1x1370x384xf32>, tensor<384xf32>, none) -> (tensor<1x1370x384xf32>, none, none)
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_Y_]], [[PARAM_2_]]) : (tensor<1x1370x384xf32>, tensor<384xf32>) -> tensor<1x1370x384xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_Y_]], [[PARAM_3_]]) : (tensor<1x1370x384xf32>, tensor<384xf32>) -> tensor<1x1370x384xf32>
// CHECK:           return [[VAR_3_]], [[VAR_4_]] : tensor<1x1370x384xf32>, tensor<1x1370x384xf32>
// CHECK:         }
}

// -----

// RMS Layer norm (sub switched)

func.func @rms_layer_norm_v1(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %d = "onnx.Sub"(%mean, %x) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x384x768xf32>, tensor<1xi64>) -> tensor<1x384x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[VAR_1_]], [[PARAM_0_]]) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_2_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm

func.func @rms_layer_norm_v2(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %dd = "onnx.Mul"(%x, %x) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%x, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %Y = "onnx.Add"(%bias, %NormScaled) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm (containing pow(varEps, -0.5))

func.func @rms_layer_norm_v3(%x: tensor<1x384x768xf32>) -> (tensor<1x384x768xf32>) {
  %neg_half = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %xx = "onnx.Mul"(%x, %x) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%xx) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %invStdDev = "onnx.Pow"(%varEps, %neg_half) : (tensor<1x384x1xf32>, tensor<f32>) -> tensor<1x384x1xf32>
  %Y = "onnx.Mul"(%invStdDev, %x) : (tensor<1x384x1xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %Y : tensor<1x384x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>) -> tensor<1x384x768xf32> {
// CHECK-DAG:       [[PARAM_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<768xf32>
// CHECK-DAG:       [[PARAM_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm (containing pow(varEps, -0.5))

func.func @rms_layer_norm_v3_dyn_shape(%x: tensor<1x?x768xf32>) -> (tensor<1x?x768xf32>) {
  %neg_half = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %xx = "onnx.Mul"(%x, %x) : (tensor<1x?x768xf32>, tensor<1x?x768xf32>) -> tensor<1x?x768xf32>
  %var = "onnx.ReduceMeanV13"(%xx) {axes = [-1], keepdims = 1 : si64, onnx_node_name = "ReduceMean_42"} : (tensor<1x?x768xf32>) -> tensor<1x?x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
  %invStdDev = "onnx.Pow"(%varEps, %neg_half) : (tensor<1x?x1xf32>, tensor<f32>) -> tensor<1x?x1xf32>
  %Y = "onnx.Mul"(%invStdDev, %x) : (tensor<1x?x1xf32>, tensor<1x?x768xf32>) -> tensor<1x?x768xf32>
  return %Y : tensor<1x?x768xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_v3_dyn_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x768xf32>) -> tensor<1x?x768xf32> {
// CHECK-DAG:       [[PARAM_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<768xf32>
// CHECK-DAG:       [[PARAM_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x?x768xf32>, tensor<768xf32>, none) -> (tensor<1x?x768xf32>, none)
// CHECK:           return [[Y_]] : tensor<1x?x768xf32>
// CHECK:         }
}

// -----

// RMS Layer norm with multiple uses of the scale multiplication

func.func @rms_layer_norm_multi_use(%x: tensor<1x384x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x384x768xf32>) {
  %eps = onnx.Constant dense<1.2E+0> : tensor<f32>
  %dd = "onnx.Mul"(%x, %x) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x384x768xf32>) -> tensor<1x384x1xf32>
  %varEps = "onnx.Add"(%eps, %var) : (tensor<f32>, tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x384x1xf32>) -> tensor<1x384x1xf32>
  %Norm = "onnx.Div"(%x, %StdDev) : (tensor<1x384x768xf32>, tensor<1x384x1xf32>) -> tensor<1x384x768xf32>
  %NormScaled = "onnx.Mul"(%scale, %Norm) : (tensor<768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  %MultiUse = "onnx.Add"(%NormScaled, %NormScaled) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
  return %MultiUse : tensor<1x384x768xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @rms_layer_norm_multi_use
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x384x768xf32>, [[PARAM_1_:%.+]]: tensor<768xf32>, [[PARAM_2_:%.+]]: tensor<768xf32>) -> tensor<1x384x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = 2 : si64, epsilon = 1.200000e+00 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none)
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_Y_]], [[VAR_Y_]]) : (tensor<1x384x768xf32>, tensor<1x384x768xf32>) -> tensor<1x384x768xf32>
// CHECK:           return [[VAR_1_]] : tensor<1x384x768xf32>
// CHECK:         }
}

// -----

// COM: QLinearMatMul
func.func @qlinear_matmul(%arg0: tensor<?x?x768xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>, %arg3: tensor<768x768xi8>, %arg4: tensor<f32>, %arg5: tensor<i8>, %arg6: tensor<f32>, %arg7: tensor<i8>) -> (tensor<?x?x768xi8>) {
    %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xf32>
    %1 = "onnx.DequantizeLinear"(%arg3, %arg4, %arg5) {axis = 1 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
    %2 = "onnx.MatMul"(%0, %1) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %3 = "onnx.QuantizeLinear"(%2, %arg6, %arg7) {axis = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
    return %3: tensor<?x?x768xi8>

// COM: AMD Disabled
// DISABLED-LABEL:  func.func @qlinear_matmul
// DISABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xi8>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<i8>, [[PARAM_3_:%.+]]: tensor<768x768xi8>, [[PARAM_4_:%.+]]: tensor<f32>, [[PARAM_5_:%.+]]: tensor<i8>, [[PARAM_6_:%.+]]: tensor<f32>, [[PARAM_7_:%.+]]: tensor<i8>) -> tensor<?x?x768xi8> {
// DISABLED:           [[VAR_0_:%.+]] = "onnx.QLinearMatMul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]]) : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>, tensor<768x768xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xi8>
// DISABLED:           return [[VAR_0_]] : tensor<?x?x768xi8>
// DISABLED:         }
}

// -----


func.func @qlinear_matmul_with_result_type(%arg0: tensor<?x?x768xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>, %arg3: tensor<768x768xi8>, %arg4: tensor<f32>, %arg5: tensor<i8>, %arg6: tensor<f32>, %arg7: tensor<i8>) -> (tensor<1x2x768xi8>) {
    %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x768xf32>
    %1 = "onnx.DequantizeLinear"(%arg3, %arg4, %arg5) {axis = 1 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
    %2 = "onnx.MatMul"(%0, %1) : (tensor<?x?x768xf32>, tensor<768x768xf32>) -> tensor<?x?x768xf32>
    %3 = "onnx.QuantizeLinear"(%2, %arg6, %arg7) {axis = 1 : si64} : (tensor<?x?x768xf32>, tensor<f32>, tensor<i8>) -> tensor<1x2x768xi8>
    return %3: tensor<1x2x768xi8>
// COM: AMD Disabled
// DISABLED-LABEL:  func.func @qlinear_matmul_with_result_type
// DISABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x768xi8>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<i8>, [[PARAM_3_:%.+]]: tensor<768x768xi8>, [[PARAM_4_:%.+]]: tensor<f32>, [[PARAM_5_:%.+]]: tensor<i8>, [[PARAM_6_:%.+]]: tensor<f32>, [[PARAM_7_:%.+]]: tensor<i8>) -> tensor<1x2x768xi8> {
// DISABLED:           [[VAR_0_:%.+]] = "onnx.QLinearMatMul"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[PARAM_7_]]) : (tensor<?x?x768xi8>, tensor<f32>, tensor<i8>, tensor<768x768xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<1x2x768xi8>
// DISABLED:           return [[VAR_0_]] : tensor<1x2x768xi8>
// DISABLED:         }
}

// -----

// gelu(x) = [x * (erf(x/1.41421354) + 1)] * 0.5
func.func @test_gelu_erf_cst_1(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----


func.func @test_gelu_with_result_type(%arg0 : tensor<?x?x3072xf32>) -> tensor<1x2x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<1x2x3072xf32>
  "func.return"(%4) : (tensor<1x2x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_with_result_type
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<1x2x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<1x2x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<1x2x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) = [x * (1 + erf(x/1.41421354))] * 0.5
func.func @test_gelu_erf_cst_change_add_operand_order(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%one, %1) : (tensor<f32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_change_add_operand_order
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) = [(erf(x/1.41421354) + 1) * x] * 0.5
func.func @test_gelu_erf_cst_change_mul_operand_order_1(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%2, %arg0) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %half) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_change_mul_operand_order_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) =  0.5 * [x * (erf(x/1.41421354) + 1) * x]
func.func @test_gelu_erf_cst_change_mul_operand_order_2(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>{
  %sqrt2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %sqrt2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Erf"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Add"(%1, %one) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%arg0, %2) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%half, %3) : (tensor<f32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  "func.return"(%4) : (tensor<?x?x3072xf32>) -> ()

// CHECK-LABEL:  func.func @test_gelu_erf_cst_change_mul_operand_order_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// gelu(x) = x * (0.5 * (1 + tanh[0.797884583 * (x + 0.044715 * x^3)]))
func.func @test_gelu_tanh(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %three = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %half = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %sqrt2pi = onnx.Constant dense<0.797884583> : tensor<f32>
  %cst044715 = onnx.Constant dense<4.471500e-02> : tensor<f32>
  %0 = "onnx.Pow"(%arg0, %three) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
  %1 = "onnx.Mul"(%cst044715, %0) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = "onnx.Add"(%arg0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.Mul"(%sqrt2pi, %2) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %4 = "onnx.Tanh"(%3) : (tensor<*xf32>) -> tensor<*xf32>
  %5 = "onnx.Add"(%one, %4) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %6 = "onnx.Mul"(%half, %5) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %7 = "onnx.Mul"(%arg0, %6) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %7 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_gelu_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gelu"([[PARAM_0_]]) {approximate = "tanh"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_gelu_erf_two_adds(%arg0: tensor<?x?x3072xf32>, %arg1: tensor<3072x768xf32>) -> tensor<?x?x768xf32> {
  %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = onnx.Constant dense<1.41421354> : tensor<f32>
  %3 = onnx.Constant dense<3.000000e-01> : tensor<3072xf32>
  %4 = "onnx.Add"(%arg0, %3) : (tensor<?x?x3072xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>
  %5 = "onnx.Div"(%4, %2) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %6 = "onnx.Erf"(%5) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %7 = "onnx.Add"(%6, %1) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %8 = "onnx.Mul"(%4, %7) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %9 = "onnx.Mul"(%8, %0) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %10 = "onnx.MatMul"(%9, %arg1) : (tensor<?x?x3072xf32>, tensor<3072x768xf32>) -> tensor<?x?x768xf32>
  return %10 : tensor<?x?x768xf32>
}
// CHECK-LABEL:  func.func @test_gelu_erf_two_adds
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>, [[PARAM_1_:%.+]]: tensor<3072x768xf32>) -> tensor<?x?x768xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<3.000000e-01> : tensor<3072xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<?x?x3072xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Gelu"([[VAR_1_]]) {approximate = "none"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.MatMul"([[VAR_2_]], [[PARAM_1_]]) : (tensor<?x?x3072xf32>, tensor<3072x768xf32>) -> tensor<?x?x768xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x768xf32>
// CHECK:         }

// -----

func.func @test_depth_to_space_CRD(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-LABEL:func.func @test_depth_to_space_CRD
// CHECK-SAME:   (%[[PARAM_1:.+]]: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  %[[DTS:.+]] = "onnx.DepthToSpace"(%[[PARAM_1]]) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  return %[[DTS]] : tensor<1x32x1080x1920xf32>
//      CHECK:}

// -----

func.func @test_depth_to_space_CRD_missing_transpose_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_unexpected_first_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x540x3840xf32> {
  %0 = onnx.Constant dense<[-1, 32, 1, 4, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 524, 3840]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x1x4x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x1x4x540x960xf32>) -> tensor<1x32x540x1x960x4xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x1x960x4xf32>, tensor<4xi64>) -> tensor<1x32x540x3840xf32>
  return %4 : tensor<1x32x540x3840xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_unexpected_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 3, 5, 2]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_unexpected_second_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 1, 32, 1080, 1920]> : tensor<5xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<5xi64>) -> tensor<1x1x32x1080x1920xf32>
  return %4 : tensor<1x1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_CRD_not_static_shapes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<6xi64>) -> tensor<*xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-LABEL:func.func @test_depth_to_space_DCR
// CHECK-SAME:   (%[[PARAM_1:.+]]: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  %[[DTS:.+]] = "onnx.DepthToSpace"(%[[PARAM_1]]) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32>
//      CHECK:  return %[[DTS]] : tensor<1x32x1080x1920xf32>
//      CHECK:}

// -----

func.func @test_depth_to_space_DCR_missing_transpose_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_unexpected_first_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x2x17280x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 32, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x32x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x32x2x540x960xf32>) -> tensor<1x2x540x32x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x2x540x32x960x2xf32>, tensor<4xi64>) -> tensor<1x2x17280x1920xf32>
  return %4 : tensor<1x2x17280x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_unexpected_perm(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 2, 5, 1]} : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_unexpected_second_reshape_result(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x540x3680xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 540, 3680]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x2x2x32x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<1x2x2x32x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x540x3680xf32>
  return %4 : tensor<1x32x540x3680xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

func.func @test_depth_to_space_DCR_not_static_shapes(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[-1, 2, 2, 32, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<6xi64>) -> tensor<*xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 4, 1, 5, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<*xf32>, tensor<4xi64>) -> tensor<*xf32>
  return %4 : tensor<*xf32>
}
// CHECK-NOT: onnx.DepthToSpace

// -----

// HardSigmoid(x) = clip(x * a + b, 0, 1) with alpha=1/6, beta=0.5 in f32
func.func @test_hardsigmoid_clip_mul_add_f32(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %b = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Add"(%0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_clip_mul_add_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

func.func @test_hardsigmoid_clip_mul_add_f32_dynamic_min(%arg0 : tensor<?x?x3072xf32>, %arg1 : tensor<f32>) -> tensor<?x?x3072xf32> {
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %b = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Add"(%0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Clip"(%1, %arg1, %one) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_clip_mul_add_f32_dynamic_min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<?x?x3072xf32> {
// CHECK-DAG:           [[VAR_A_:%.+]] = onnx.Constant dense<0.166666672> : tensor<f32>
// CHECK-DAG:           [[VAR_B_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:           [[VAR_ONE_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_A_]]) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_B_]]) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
// CHECK-NOT:       "onnx.HardSigmoid"
// CHECK:           [[VAR_2_:%.+]] = "onnx.Clip"([[VAR_1_]], [[PARAM_1_]], [[VAR_ONE_]]) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_2_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// HardSigmoid(x) = clip(x * a + b, 0, 1) with alpha=1/6, beta=0.5 in bf16
func.func @test_hardsigmoid_clip_mul_add_bf16(%arg0 : tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16> {
  %a = onnx.Constant dense<0.166015625> : tensor<bf16>
  %b = onnx.Constant dense<5.000000e-01> : tensor<bf16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<bf16>
  %one = onnx.Constant dense<1.000000e+00> : tensor<bf16>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<?x?x3072xbf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  %1 = "onnx.Add"(%0, %b) : (tensor<?x?x3072xbf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<?x?x3072xbf16>, tensor<bf16>, tensor<bf16>) -> tensor<?x?x3072xbf16>
  return %2 : tensor<?x?x3072xbf16>

// CHECK-LABEL:  func.func @test_hardsigmoid_clip_mul_add_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166015625 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xbf16>) -> tensor<?x?x3072xbf16>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xbf16>
// CHECK:         }
}

// -----

// HardSigmoid(x) = clip(x * a + b, 0, 1) with alpha=1/6, beta=0.5 in f16
func.func @test_hardsigmoid_clip_mul_add_f16(%arg0 : tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16> {
  %a = onnx.Constant dense<0.166625977> : tensor<f16>
  %b = onnx.Constant dense<5.000000e-01> : tensor<f16>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f16>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f16>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<?x?x3072xf16>, tensor<f16>) -> tensor<?x?x3072xf16>
  %1 = "onnx.Add"(%0, %b) : (tensor<?x?x3072xf16>, tensor<f16>) -> tensor<?x?x3072xf16>
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<?x?x3072xf16>, tensor<f16>, tensor<f16>) -> tensor<?x?x3072xf16>
  return %2 : tensor<?x?x3072xf16>

// CHECK-LABEL:  func.func @test_hardsigmoid_clip_mul_add_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166625977 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf16>) -> tensor<?x?x3072xf16>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf16>
// CHECK:         }
}

// -----

// HardSigmoid(x) = clip(x * a + b, 0, 1) with alpha=0.2, beta=0.5 in f32
func.func @test_hardsigmoid_clip_mul_add_default_alpha_beta(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %a = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %b = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Add"(%0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Clip"(%1, %zero, %one) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_clip_mul_add_default_alpha_beta
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// ============================================================================
// RecomposeHardSigmoidFromReluPattern / RecomposeHardSigmoidFromClipMulPattern
//
// TF/Keras Relu6-style decompositions of HardSigmoid: [optional Add(x, ~b')]
// -> [Relu | Clip(0, ~c)] -> [optional Min(., ~c)] -> Mul(., ~a'), which is
// algebraically equivalent to canonical HardSigmoid with alpha=a',
// beta=b'*a' (pulling the scale out of the clip: clip(a*z+b,0,1) ==
// a*clip(z+b/a,0,1/a) for a>0).
// ============================================================================

// Add(x, 3) -> Relu -> Min(., 6) -> Mul(., 1/6): Relu+Min together are
// exactly Clip(0, 6), so this is unconditionally safe (Constraint A).
// CHECK-LABEL: @test_hardsigmoid_relu_min_pass
func.func @test_hardsigmoid_relu_min_pass(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %b = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %c = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Relu"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Min"(%1, %c) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %3 = "onnx.Mul"(%2, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %3 : tensor<?x?x3072xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// Relu -> Min(., 4) -> Mul(., 0.25), with no leading Add at all: the "+beta"
// term is not present as its own node (e.g. folded into a preceding op's
// bias upstream), so beta defaults to 0. Also proves alpha/beta are derived
// generically rather than restricted to the canonical 1/6, 0.5 pair.
// CHECK-LABEL: @test_hardsigmoid_relu_min_no_add_pass
func.func @test_hardsigmoid_relu_min_no_add_pass(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %c = onnx.Constant dense<4.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %0 = "onnx.Relu"(%arg0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Min"(%0, %c) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 2.500000e-01 : f32, beta = 0.000000e+00 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// Add(x, 3) -> Relu -> QuantizeLinear -> DequantizeLinear -> Mul(., 1/6), no
// explicit Min. The quantizer's own saturation (scale chosen so
// representable_max == 127*6/127 == 6) already enforces the missing upper
// clamp, so the bare Relu is provably safe (Constraint B) even though it
// only implements a one-sided clamp on its own.
// CHECK-LABEL: @test_hardsigmoid_relu_quant_safe_pass
func.func @test_hardsigmoid_relu_quant_safe_pass(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %b = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %scale = onnx.Constant dense<0.0472440943> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %0 = "onnx.Add"(%arg0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Relu"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.QuantizeLinear"(%1, %scale, %zp) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x3072xi8>
  %3 = "onnx.DequantizeLinear"(%2, %scale, %zp) : (tensor<?x?x3072xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %4 : tensor<?x?x3072xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// Same shape as test_hardsigmoid_relu_quant_safe_pass, but the quantization
// scale is too loose: representable_max (127*0.06 ~= 7.62) exceeds the
// algebraic bound 1/alpha' (6), so the quantizer's saturation does not
// prove the missing upper clamp is dead code. Must be rejected.
func.func @test_hardsigmoid_relu_quant_unsafe_reject(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %b = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %scale = onnx.Constant dense<6.000000e-02> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<i8>
  %0 = "onnx.Add"(%arg0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Relu"(%0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.QuantizeLinear"(%1, %scale, %zp) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<i8>) -> tensor<?x?x3072xi8>
  %3 = "onnx.DequantizeLinear"(%2, %scale, %zp) : (tensor<?x?x3072xi8>, tensor<f32>, tensor<i8>) -> tensor<?x?x3072xf32>
  %4 = "onnx.Mul"(%3, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %4 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_relu_quant_unsafe_reject
// CHECK-NOT:       "onnx.HardSigmoid"
// CHECK:           "onnx.Relu"
// CHECK:           "onnx.QuantizeLinear"
// CHECK:           "onnx.DequantizeLinear"
// CHECK:           "onnx.Mul"
}

// -----

// Bare Relu(x) -> Mul(., 1/6), pure float graph: no explicit Min and no
// adjoining Quantize/Dequantize pair, so neither Constraint A nor B can
// prove the missing upper clamp is safe. Must be rejected.
func.func @test_hardsigmoid_relu_no_proof_reject(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %0 = "onnx.Relu"(%arg0) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Mul"(%0, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %1 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_relu_no_proof_reject
// CHECK-NOT:       "onnx.HardSigmoid"
// CHECK:           "onnx.Relu"
// CHECK:           "onnx.Mul"
}

// -----

// Add(x, 3) -> Clip(0, 6) -> Mul(., 1/6): TF-style reordering of the
// canonical explicit-Clip HardSigmoid, where the scale is applied *after*
// the clip rather than before it. Clip's own bounds already supply the
// two-sided clamp, so no quantization safety proof is needed here.
// CHECK-LABEL: @test_hardsigmoid_clip_mul_tf_style_pass
func.func @test_hardsigmoid_clip_mul_tf_style_pass(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %b = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %c = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Clip"(%0, %zero, %c) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// Clip(0, 4) -> Mul(., 0.25), with no leading Add: same "+beta absorbed
// upstream" case as test_hardsigmoid_relu_min_no_add_pass, but through the
// Clip-anchored (RecomposeHardSigmoidFromClipMulPattern) path instead of
// the Relu-anchored one.
// CHECK-LABEL: @test_hardsigmoid_clip_mul_tf_style_no_add_pass
func.func @test_hardsigmoid_clip_mul_tf_style_no_add_pass(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %c = onnx.Constant dense<4.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %0 = "onnx.Clip"(%arg0, %zero, %c) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Mul"(%0, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %1 : tensor<?x?x3072xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 2.500000e-01 : f32, beta = 0.000000e+00 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// Add(x, 3) -> Clip(0, 5) -> Mul(., 1/6): Clip's upper bound (5) is not
// ~1/alpha' (6) as required for Clip(0,c) to equal clip(a'*z, 0, 1) once the
// scale is pulled back out, so this must be rejected rather than silently
// recomposed into an incorrect HardSigmoid.
func.func @test_hardsigmoid_clip_mul_tf_style_bound_mismatch_reject(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %b = onnx.Constant dense<3.000000e+00> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %c = onnx.Constant dense<5.000000e+00> : tensor<f32>
  %a = onnx.Constant dense<0.166666672> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %b) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Clip"(%0, %zero, %c) : (tensor<?x?x3072xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Mul"(%1, %a) : (tensor<?x?x3072xf32>, tensor<f32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardsigmoid_clip_mul_tf_style_bound_mismatch_reject
// CHECK-NOT:       "onnx.HardSigmoid"
// CHECK:           "onnx.Add"
// CHECK:           "onnx.Clip"
// CHECK:           "onnx.Mul"
}

// -----

// HardSwish(x) = x * HardSigmoid(x) with alpha=1/6, beta=0.5
func.func @test_hardswish_from_mul_hardsigmoid(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Mul"(%arg0, %0) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  return %1 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardswish_from_mul_hardsigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK-NOT:       "onnx.HardSigmoid"
// CHECK-NOT:       "onnx.Mul"
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSwish"([[PARAM_0_]]) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

// Mul(x, HardSigmoid(x)) with alpha=0.2 should not match HardSwish pattern
func.func @test_hardswish_from_mul_hardsigmoid_wrong_alpha(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Mul"(%arg0, %0) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  return %1 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardswish_from_mul_hardsigmoid_wrong_alpha
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK-NOT:       "onnx.HardSwish"
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_1_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

func.func @test_hardswish_from_mul_hardsigmoid_wrong_beta(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 3.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %1 = "onnx.Mul"(%arg0, %0) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  return %1 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardswish_from_mul_hardsigmoid_wrong_beta
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
// CHECK-NOT:       "onnx.HardSwish"
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166666672 : f32, beta = 3.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_0_]]) : (tensor<?x?x3072xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
// CHECK:           return [[VAR_1_]] : tensor<?x?x3072xf32>
// CHECK:         }
}

// -----

func.func @test_hardswish_from_mul_hardsigmoid_wrong_input(%arg0 : tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32> {
  %0 = "onnx.Constant"() {value=dense<[7.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  %2 = "onnx.Mul"(%0, %1) : (tensor<1xf32>, tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
  return %2 : tensor<?x?x3072xf32>

// CHECK-LABEL:  func.func @test_hardswish_from_mul_hardsigmoid_wrong_input
// CHECK-NOT:       "onnx.HardSwish"
// CHECK:           "onnx.HardSigmoid"
// CHECK:           "onnx.Mul"
// CHECK:         }
}
