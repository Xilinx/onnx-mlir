// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @gemm_to_fc(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:  func.func @gemm_to_fc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 4, 5>} : (tensor<4x5xf32>) -> tensor<1x4x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_2_]] : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_3_]] : (tensor<1x1x5xf32>, tensor<1x5x4xf32>) -> tensor<1x1x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 1, 4>} : (tensor<4xf32>) -> tensor<1x1x4xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x1x4xf32>, tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]] {new_shape = array<i64: 1, 4>} : (tensor<1x1x4xf32>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x4xf32>
// CHECK:         }
}
  
// -----
  
func.func @gemm_to_fc_broadcast(%arg0: tensor<2x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<2x5xf32>, tensor<4x5xf32>, tensor<1xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
// CHECK-LABEL:  func.func @gemm_to_fc_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>) -> tensor<2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 2, 5>} : (tensor<2x5xf32>) -> tensor<1x2x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 4, 5>} : (tensor<4x5xf32>) -> tensor<1x4x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_2_]] : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_3_]] : (tensor<1x2x5xf32>, tensor<1x5x4xf32>) -> tensor<1x2x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x2x4xf32>, tensor<1x1x1xf32>) -> tensor<1x2x4xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]] {new_shape = array<i64: 2, 4>} : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<2x4xf32>
// CHECK:         }
}

// -----

func.func @gemm_to_fc_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:  func.func @gemm_to_fc_opt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>) -> tensor<1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, 1, 5>} : (tensor<1x5xf32>) -> tensor<1x1x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 4, 5>} : (tensor<4x5xf32>) -> tensor<1x4x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<1x4x5xf32>, tensor<3xi32>) -> tensor<1x5x4xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_4_]] : (tensor<1x1x5xf32>, tensor<1x5x4xf32>) -> tensor<1x1x4xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_5_]] {new_shape = array<i64: 1, 4>} : (tensor<1x1x4xf32>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x4xf32>
// CHECK:         }
}

// -----

func.func @gemm_to_fc_dynamic_input_static_output(%arg0: tensor<?x5xf32>, %arg1: tensor<4x?xf32>, %arg2: tensor<?xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<?x5xf32>, tensor<4x?xf32>, tensor<?xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
// CHECK-LABEL:  func.func @gemm_to_fc_dynamic_input_static_output
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x5xf32>, [[PARAM_1_:%.+]]: tensor<4x?xf32>, [[PARAM_2_:%.+]]: tensor<?xf32>) -> tensor<1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reshape [[PARAM_0_]] {new_shape = array<i64: 1, -9223372036854775808, 5>} : (tensor<?x5xf32>) -> tensor<1x?x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 4, -9223372036854775808>} : (tensor<4x?xf32>) -> tensor<1x4x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_3_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_2_]] : (tensor<1x4x?xf32>, tensor<3xi32>) -> tensor<1x?x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_0_]], [[VAR_3_]] : (tensor<1x?x5xf32>, tensor<1x?x4xf32>) -> tensor<1x?x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 1, -9223372036854775808>} : (tensor<?xf32>) -> tensor<1x1x?xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.add [[VAR_4_]], [[VAR_5_]] : (tensor<1x?x4xf32>, tensor<1x1x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_6_]] {new_shape = array<i64: 1, 4>} : (tensor<?x?x?xf32>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x4xf32>
// CHECK:         }
}
