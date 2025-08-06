// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{.}}[0, 1], [1, 2]{{.}}> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.add [[VAR_0_]], [[VAR_1_]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           [[VAR_4_:%.+]] = tosa.greater_equal [[VAR_0_]], [[VAR_3_]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi1>
// CHECK:           [[VAR_5_:%.+]] = tosa.select [[VAR_4_]], [[VAR_0_]], [[VAR_2_]] : (tensor<2x2xi1>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.cast [[VAR_5_]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_7_]] : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[1, 3, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.reshape [[VAR_8_]], [[VAR_9_]] : (tensor<3x2xf32>, !tosa.shape<3>) -> tensor<1x3x2xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_12_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_11_]] : (tensor<2x2xi32>, !tosa.shape<2>) -> tensor<1x4xi32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.gather [[VAR_10_]], [[VAR_12_]] : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.const_shape  {value = dense<2> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_15_:%.+]] = tosa.reshape [[VAR_13_]], [[VAR_14_]] : (tensor<1x4x2xf32>, !tosa.shape<3>) -> tensor<2x2x2xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "tosa.const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_17_:%.+]] = tosa.transpose [[VAR_15_]], [[VAR_16_]] : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK:           return [[VAR_17_]] : tensor<2x2x2xf32>
}

// -----

// Test negative indices.
func.func @test_gather_axis0_neg_idx(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_axis0_neg_idx
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x2xf32>) -> tensor<2x2x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.add [[VAR_0_]], [[VAR_1_]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           [[VAR_4_:%.+]] = tosa.greater_equal [[VAR_0_]], [[VAR_3_]] : (tensor<2x2xi64>, tensor<1x1xi64>) -> tensor<2x2xi1>
// CHECK:           [[VAR_5_:%.+]] = tosa.select [[VAR_4_]], [[VAR_0_]], [[VAR_2_]] : (tensor<2x2xi1>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.cast [[VAR_5_]] : (tensor<2x2xi64>) -> tensor<2x2xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_7_]] : (tensor<3x2xf32>, tensor<2xi32>) -> tensor<3x2xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[1, 3, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.reshape [[VAR_8_]], [[VAR_9_]] : (tensor<3x2xf32>, !tosa.shape<3>) -> tensor<1x3x2xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[1, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_12_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_11_]] : (tensor<2x2xi32>, !tosa.shape<2>) -> tensor<1x4xi32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.gather [[VAR_10_]], [[VAR_12_]] : (tensor<1x3x2xf32>, tensor<1x4xi32>) -> tensor<1x4x2xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.const_shape  {value = dense<2> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_15_:%.+]] = tosa.reshape [[VAR_13_]], [[VAR_14_]] : (tensor<1x4x2xf32>, !tosa.shape<3>) -> tensor<2x2x2xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "tosa.const"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_17_:%.+]] = tosa.transpose [[VAR_15_]], [[VAR_16_]] : (tensor<2x2x2xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
// CHECK:           return [[VAR_17_]] : tensor<2x2x2xf32>
}

// -----

// Test along axis 1. Transpose should be different.
func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_axis1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<3x1x2xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<{{.}}[0, 2]{{.}}> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.add [[VAR_0_]], [[VAR_1_]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           [[VAR_4_:%.+]] = tosa.greater_equal [[VAR_0_]], [[VAR_3_]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi1>
// CHECK:           [[VAR_5_:%.+]] = tosa.select [[VAR_4_]], [[VAR_0_]], [[VAR_2_]] : (tensor<1x2xi1>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.cast [[VAR_5_]] : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_7_]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[1, 3, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.reshape [[VAR_8_]], [[VAR_9_]] : (tensor<3x3xf32>, !tosa.shape<3>) -> tensor<1x3x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_12_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_11_]] : (tensor<1x2xi32>, !tosa.shape<2>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.gather [[VAR_10_]], [[VAR_12_]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.const_shape  {value = dense<[1, 2, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_15_:%.+]] = tosa.reshape [[VAR_13_]], [[VAR_14_]] : (tensor<1x2x3xf32>, !tosa.shape<3>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_17_:%.+]] = tosa.transpose [[VAR_15_]], [[VAR_16_]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return [[VAR_17_]] : tensor<3x1x2xf32>
}

// -----

func.func @test_gather_dynamic_indices(%arg0 : tensor<3x3xf32>, %indices: tensor<1x2xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_dynamic_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2xi64>) -> tensor<3x1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.add [[PARAM_1_]], [[VAR_0_]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK:           [[VAR_3_:%.+]] = tosa.greater_equal [[PARAM_1_]], [[VAR_2_]] : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi1>
// CHECK:           [[VAR_4_:%.+]] = tosa.select [[VAR_3_]], [[PARAM_1_]], [[VAR_1_]] : (tensor<1x2xi1>, tensor<1x2xi64>, tensor<1x2xi64>) -> tensor<1x2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.cast [[VAR_4_]] : (tensor<1x2xi64>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_6_]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[1, 3, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<3x3xf32>, !tosa.shape<3>) -> tensor<1x3x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.const_shape  {value = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_11_:%.+]] = tosa.reshape [[VAR_5_]], [[VAR_10_]] : (tensor<1x2xi32>, !tosa.shape<2>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.gather [[VAR_9_]], [[VAR_11_]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.const_shape  {value = dense<[1, 2, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.reshape [[VAR_12_]], [[VAR_13_]] : (tensor<1x2x3xf32>, !tosa.shape<3>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_16_:%.+]] = tosa.transpose [[VAR_14_]], [[VAR_15_]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return [[VAR_16_]] : tensor<3x1x2xf32>
}

// -----

func.func @test_gather_dynamic_indices_i32(%arg0 : tensor<3x3xf32>, %indices: tensor<1x2xi32>) -> tensor<3x1x2xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi32>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_dynamic_indices_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>, [[PARAM_1_:%.+]]: tensor<1x2xi32>) -> tensor<3x1x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<3> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.add [[PARAM_1_]], [[VAR_0_]] : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.greater_equal [[PARAM_1_]], [[VAR_2_]] : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi1>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.select [[VAR_3_]], [[PARAM_1_]], [[VAR_1_]] : (tensor<1x2xi1>, tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_5_]] : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<[1, 3, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_7_]] : (tensor<3x3xf32>, !tosa.shape<3>) -> tensor<1x3x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_10_:%.+]] = tosa.reshape [[VAR_4_]], [[VAR_9_]] : (tensor<1x2xi32>, !tosa.shape<2>) -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.gather [[VAR_8_]], [[VAR_10_]] : (tensor<1x3x3xf32>, tensor<1x2xi32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.const_shape  {value = dense<[1, 2, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.reshape [[VAR_11_]], [[VAR_12_]] : (tensor<1x2x3xf32>, !tosa.shape<3>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_15_:%.+]] = tosa.transpose [[VAR_13_]], [[VAR_14_]] : (tensor<1x2x3xf32>, tensor<3xi32>) -> tensor<3x1x2xf32>
// CHECK:           return [[VAR_15_]] : tensor<3x1x2xf32>
}

// -----

func.func @test_gather_like_slice(%arg0 : tensor<3x3xf32>) -> tensor<3xf32> {
  %indices = onnx.Constant dense<0> : tensor<i64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<i64>) -> tensor<3xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_like_slice
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[3, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<3x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[VAR_3_]] : (tensor<3x1xf32>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK:           return [[VAR_4_]] : tensor<3xf32>
}

// -----

func.func @test_gather_like_slice_positive_integer(%arg0 : tensor<3x3xf32>) -> tensor<3xf32> {
  %indices = onnx.Constant dense<2> : tensor<i64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<i64>) -> tensor<3xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_like_slice_positive_integer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[2, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<3x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[VAR_3_]] : (tensor<1x3xf32>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK:           return [[VAR_4_]] : tensor<3xf32>
}

// -----

func.func @test_gather_like_slice_negative_integer(%arg0 : tensor<3x3xf32>) -> tensor<3xf32> {
  %indices = onnx.Constant dense<-1> : tensor<i64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<i64>) -> tensor<3xf32>
  "func.return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_like_slice_negative_integer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[2, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<3x3xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[VAR_3_]] : (tensor<1x3xf32>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK:           return [[VAR_4_]] : tensor<3xf32>
}

// -----

func.func @test_gather_dynamic_shape_indices_i32(%arg0 : tensor<?x4xf32>, %indices: tensor<?xi64>) -> tensor<?x4xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<?x4xf32>, tensor<?xi64>) -> tensor<?x4xf32>
  "func.return"(%0) : (tensor<?x4xf32>) -> ()
// CHECK-LABEL: test_gather_dynamic_shape_indices_i32
// CHECK: onnx.Gather
}
