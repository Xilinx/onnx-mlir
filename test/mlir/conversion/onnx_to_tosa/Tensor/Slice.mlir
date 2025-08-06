// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s


func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL:  func.func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<2x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x3xf32>
}

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL:  func.func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<2x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x3xf32>
}

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL:  func.func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>) -> tensor<1x3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<2x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x3xf32>
}

// -----

func.func @slice_all_dynamic(%arg0: tensor<20x10x5xf32>,
                             %arg1: tensor<1xi64>,
                             %arg2: tensor<1xi64>,
                             %arg3: tensor<1xi64>,
                             %arg4: tensor<1xi64>)
                              -> tensor<20x9x5xf32> {
  %0 = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<20x10x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<20x9x5xf32>
  return %0 : tensor<20x9x5xf32>
}
// CHECK-LABEL:  func.func @slice_all_dynamic
// CHECK: onnx.Slice

// -----

func.func @slice_dynamic_axes(%arg0: tensor<20x10x5xf32>,
                             %arg1: tensor<3xi64>,
                             %arg2: tensor<3xi64>)
                              -> tensor<20x10x1xf32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %1 = onnx.Constant dense<1> : tensor<3xi64>
  %2 = "onnx.Slice"(%arg0, %arg1, %arg2, %0, %1) {onnx_node_name = "onnx.Slice_0"} : (tensor<20x10x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<20x10x1xf32>
  return %2 : tensor<20x10x1xf32>
}
// CHECK-LABEL:  func.func @slice_dynamic_axes
// CHECK: onnx.Slice

// -----

func.func @slice_just_steps(%arg0: tensor<100x200xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[100, 200]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32> 
}
// CHECK-LABEL:  func.func @slice_just_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x200xf32>) -> tensor<20x20xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[20, 5, 20, 10]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<100x200xf32>, !tosa.shape<4>) -> tensor<20x5x20x10xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[20, 1, 20, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] : (tensor<20x5x20x10xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<20x1x20x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<20> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_4_]], [[VAR_5_]] : (tensor<20x1x20x1xf32>, !tosa.shape<2>) -> tensor<20x20xf32>
// CHECK:           return [[VAR_6_]] : tensor<20x20xf32>
// CHECK:         }

// -----

func.func @slice_steps_and_edges(%arg0: tensor<100x200xf32>) -> tensor<16x17xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[82, 178]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<16x17xf32>
  return %1 : tensor<16x17xf32> 
}
// CHECK-LABEL:  func.func @slice_steps_and_edges
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x200xf32>) -> tensor<16x17xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[5, 10]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[80, 170]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<100x200xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<80x170xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[16, 5, 17, 10]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[VAR_3_]] : (tensor<80x170xf32>, !tosa.shape<4>) -> tensor<16x5x17x10xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[16, 1, 17, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.slice [[VAR_4_]], [[VAR_5_]], [[VAR_6_]] : (tensor<16x5x17x10xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<16x1x17x1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[16, 17]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<16x1x17x1xf32>, !tosa.shape<2>) -> tensor<16x17xf32>
// CHECK:           return [[VAR_9_]] : tensor<16x17xf32>
// CHECK:         }

// -----

func.func @slice_steps_and_edges_with_padding(%arg0: tensor<99x195xf32>) -> tensor<19x19xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[97, 192]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<99x195xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<19x19xf32>
  return %1 : tensor<19x19xf32> 
}
// CHECK-LABEL:  func.func @slice_steps_and_edges_with_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<99x195xf32>) -> tensor<19x19xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[0, 1, 0, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<99x195xf32>, !tosa.shape<4>, tensor<f32>) -> tensor<100x200xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[5, 10]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[95, 190]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_3_]], [[VAR_4_]] : (tensor<100x200xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<95x190xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[19, 5, 19, 10]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_5_]], [[VAR_6_]] : (tensor<95x190xf32>, !tosa.shape<4>) -> tensor<19x5x19x10xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[19, 1, 19, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.slice [[VAR_7_]], [[VAR_8_]], [[VAR_9_]] : (tensor<19x5x19x10xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<19x1x19x1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<19> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_12_:%.+]] = tosa.reshape [[VAR_10_]], [[VAR_11_]] : (tensor<19x1x19x1xf32>, !tosa.shape<2>) -> tensor<19x19xf32>
// CHECK:           return [[VAR_12_]] : tensor<19x19xf32>
// CHECK:         }

// -----

func.func @slice_just_steps_with_padding(%arg0: tensor<99x195xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[99, 195]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<99x195xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32> 
}
// CHECK-LABEL:  func.func @slice_just_steps_with_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<99x195xf32>) -> tensor<20x20xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[0, 1, 0, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<99x195xf32>, !tosa.shape<4>, tensor<f32>) -> tensor<100x200xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[20, 5, 20, 10]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[VAR_3_]] : (tensor<100x200xf32>, !tosa.shape<4>) -> tensor<20x5x20x10xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[20, 1, 20, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.slice [[VAR_4_]], [[VAR_5_]], [[VAR_6_]] : (tensor<20x5x20x10xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<20x1x20x1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<20> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<20x1x20x1xf32>, !tosa.shape<2>) -> tensor<20x20xf32>
// CHECK:           return [[VAR_9_]] : tensor<20x20xf32>
// CHECK:         }

// -----

func.func @slice_negative_steps(%arg0: tensor<100x200xf32>) -> tensor<20x20xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[-5, -10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<100x200xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<20x20xf32>
  return %1 : tensor<20x20xf32> 
}
// CHECK-LABEL: func @slice_negative_steps
// CHECK: onnx.Slice

// -----

func.func @slice_start_greater_than_dim(%arg0: tensor<10x30xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[20, 20]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[21,21]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<1> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<10x30xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32> 
}
// CHECK-LABEL:  func.func @slice_start_greater_than_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x30xf32>) -> tensor<0x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[10, 20]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[0, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<10x30xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<0x1xf32>
// CHECK:           return [[VAR_2_]] : tensor<0x1xf32>
// CHECK:         }

// -----

func.func @slice_step_greater_than_dim(%arg0: tensor<9x30xf32>) -> tensor<1x2xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[5, 5]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[8, 25]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[10, 10]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<9x30xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32> 
}
// CHECK-LABEL:  func.func @slice_step_greater_than_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<9x30xf32>) -> tensor<1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[0, 6, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<9x30xf32>, !tosa.shape<4>, tensor<f32>) -> tensor<15x30xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<5> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[10, 20]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_3_]], [[VAR_4_]] : (tensor<15x30xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<10x20xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[1, 10, 2, 10]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_5_]], [[VAR_6_]] : (tensor<10x20xf32>, !tosa.shape<4>) -> tensor<1x10x2x10xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 2, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.slice [[VAR_7_]], [[VAR_8_]], [[VAR_9_]] : (tensor<1x10x2x10xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x1x2x1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_12_:%.+]] = tosa.reshape [[VAR_10_]], [[VAR_11_]] : (tensor<1x1x2x1xf32>, !tosa.shape<2>) -> tensor<1x2xf32>
// CHECK:           return [[VAR_12_]] : tensor<1x2xf32>
// CHECK:         }

// -----

func.func @slice_4d(%arg0: tensor<1x56x56x92xf32>) -> tensor<1x28x28x92xf32> {
  %axes = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[56,56]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[2, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<1x56x56x92xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x28x28x92xf32>
  return %1 : tensor<1x28x28x92xf32> 
}
// CHECK-LABEL:  func.func @slice_4d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x56x56x92xf32>) -> tensor<1x28x28x92xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.pad [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<1x56x56x92xf32>, !tosa.shape<8>, tensor<f32>) -> tensor<1x57x57x92xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[0, 1, 1, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[1, 56, 56, 92]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_3_]], [[VAR_4_]] : (tensor<1x57x57x92xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x56x56x92xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[1, 28, 2, 28, 2, 92]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.reshape [[VAR_5_]], [[VAR_6_]] : (tensor<1x56x56x92xf32>, !tosa.shape<6>) -> tensor<1x28x2x28x2x92xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[1, 28, 1, 28, 1, 92]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.slice [[VAR_7_]], [[VAR_8_]], [[VAR_9_]] : (tensor<1x28x2x28x2x92xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<1x28x1x28x1x92xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[1, 28, 28, 92]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_12_:%.+]] = tosa.reshape [[VAR_10_]], [[VAR_11_]] : (tensor<1x28x1x28x1x92xf32>, !tosa.shape<4>) -> tensor<1x28x28x92xf32>
// CHECK:           return [[VAR_12_]] : tensor<1x28x28x92xf32>
// CHECK:         }
