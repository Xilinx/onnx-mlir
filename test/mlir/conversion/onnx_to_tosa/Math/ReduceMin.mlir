// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @reduce_min(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMin"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_min(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_min %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_min %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x5x1x1xf32>
}

// -----

func.func @reduce_min_no_axes_attr(%arg0: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMin"(%arg0, %none) : (tensor<2x5x9x11xf32>, none) -> tensor<1x1x1x1xf32>
return %0 : tensor<1x1x1x1xf32>
// CHECK-LABEL:   func.func @reduce_min_no_axes_attr(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_min %[[VAL_0]] {axis = 0 : i32} : (tensor<2x5x9x11xf32>) -> tensor<1x5x9x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_min %[[VAL_1]] {axis = 1 : i32} : (tensor<1x5x9x11xf32>) -> tensor<1x1x9x11xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reduce_min %[[VAL_2]] {axis = 2 : i32} : (tensor<1x1x9x11xf32>) -> tensor<1x1x1x11xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.reduce_min %[[VAL_3]] {axis = 3 : i32} : (tensor<1x1x1x11xf32>) -> tensor<1x1x1x1xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x1x1x1xf32>
}

// -----

func.func @reduce_min_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMin"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
// CHECK-LABEL:   func.func @reduce_min_keepdims_false(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_min %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_min %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[2, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape %[[VAL_2]], [[VAR_2_]] : (tensor<2x5x1x1xf32>, !tosa.shape<2>) -> tensor<2x5xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x5xf32>
}

// -----

func.func @reduce_min_noop_with_emtpy_axes_one(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMin"(%arg0, %0) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_min_noop_with_emtpy_axes_one(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_min %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_min %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x5x1x1xf32>
}

// -----

func.func @reduce_min_noop_with_emtpy_axes_one_none_input(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMin"(%arg0, %none) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, none) ->  tensor<2x5x9x11xf32>
return %0 : tensor<2x5x9x11xf32>
// CHECK-LABEL:   func.func @reduce_min_noop_with_emtpy_axes_one_none_input(
// CHECK-SAME:                                                               %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.identity %[[VAL_0]] : (tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x5x9x11xf32>
}

// -----

func.func @test_reduceminV13(%arg0: tensor<1x32x112x112xf32>) -> tensor<1x32x1x1xf32> {
  %0 = "onnx.ReduceMinV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x32x112x112xf32>) -> tensor<1x32x1x1xf32>
  return %0 : tensor<1x32x1x1xf32>
// CHECK-LABEL:  func.func @test_reduceminV13
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_min %arg0 {axis = 2 : i32}
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reduce_min [[VAR_0_]] {axis = 3 : i32}
// CHECK:           return [[VAR_1_]] : tensor<1x32x1x1xf32>
}

// -----

func.func @test_reduceminV13_keep_dims_0(%arg0: tensor<1x32x112x112xf32>) -> tensor<1x32xf32> {
  %0 = "onnx.ReduceMinV13"(%arg0) {axes = [2, 3], keepdims = 0 : si64} : (tensor<1x32x112x112xf32>) -> tensor<1x32xf32>
  return %0 : tensor<1x32xf32>
// CHECK-LABEL:  func.func @test_reduceminV13_keep_dims_0
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_min %arg0 {axis = 2 : i32}
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reduce_min [[VAR_0_]] {axis = 3 : i32}
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 32]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_2_]] : (tensor<1x32x1x1xf32>, !tosa.shape<2>) -> tensor<1x32xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x32xf32>
}
