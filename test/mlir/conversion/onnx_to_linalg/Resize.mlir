// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Advanced Micro Devices, Inc.

// RUN: onnx-mlir-opt --convert-onnx-resize-to-linalg %s -split-input-file | FileCheck %s
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:   func.func @nearest_asymmetric_scales(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x2x4x4xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_3]] : tensor<1x2x4x4xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32):
// CHECK:             %[[VAL_6:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i64
// CHECK:             %[[VAL_8:.*]] = arith.sitofp %[[VAL_7]] : i64 to f32
// CHECK:             %[[VAL_9:.*]] = math.floor %[[VAL_8]] : f32
// CHECK:             %[[VAL_10:.*]] = arith.fptosi %[[VAL_9]] : f32 to i64
// CHECK:             %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i64 to index
// CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_2]] : index
// CHECK:             %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_2]], %[[VAL_11]] : index
// CHECK:             %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_1]] : index
// CHECK:             %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_1]], %[[VAL_13]] : index
// CHECK:             %[[VAL_16:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : index to i64
// CHECK:             %[[VAL_18:.*]] = arith.sitofp %[[VAL_17]] : i64 to f32
// CHECK:             %[[VAL_19:.*]] = arith.divf %[[VAL_18]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_20:.*]] = math.floor %[[VAL_19]] : f32
// CHECK:             %[[VAL_21:.*]] = arith.fptosi %[[VAL_20]] : f32 to i64
// CHECK:             %[[VAL_22:.*]] = arith.index_cast %[[VAL_21]] : i64 to index
// CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_2]] : index
// CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_2]], %[[VAL_22]] : index
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_1]] : index
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_1]], %[[VAL_24]] : index
// CHECK:             %[[VAL_27:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_28:.*]] = arith.index_cast %[[VAL_27]] : index to i64
// CHECK:             %[[VAL_29:.*]] = arith.sitofp %[[VAL_28]] : i64 to f32
// CHECK:             %[[VAL_30:.*]] = arith.divf %[[VAL_29]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_31:.*]] = math.floor %[[VAL_30]] : f32
// CHECK:             %[[VAL_32:.*]] = arith.fptosi %[[VAL_31]] : f32 to i64
// CHECK:             %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : i64 to index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_2]] : index
// CHECK:             %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_2]], %[[VAL_33]] : index
// CHECK:             %[[VAL_36:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_1]] : index
// CHECK:             %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_1]], %[[VAL_35]] : index
// CHECK:             %[[VAL_38:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_2]], %[[VAL_15]], %[[VAL_26]], %[[VAL_37]]] : tensor<1x2x2x2xf32>
// CHECK:             linalg.yield %[[VAL_38]] : f32
// CHECK:           } -> tensor<1x2x4x4xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x2x4x4xf32>
// CHECK:         }
func.func @nearest_asymmetric_scales(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %resize = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<1x2x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x2x4x4xf32>
  return %resize : tensor<1x2x4x4xf32>
}
// -----
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:   func.func @nearest_round_prefer_floor(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x2x2xi8>) -> tensor<1x1x4x4xi8> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<1x1x4x4xi8>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_4]] : tensor<1x1x4x4xi8>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i8):
// CHECK:             %[[VAL_7:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : index to i64
// CHECK:             %[[VAL_9:.*]] = arith.sitofp %[[VAL_8]] : i64 to f32
// CHECK:             %[[VAL_10:.*]] = arith.addf %[[VAL_9]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_11:.*]] = arith.divf %[[VAL_10]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_12:.*]] = arith.subf %[[VAL_11]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_13:.*]] = math.floor %[[VAL_12]] : f32
// CHECK:             %[[VAL_14:.*]] = arith.fptosi %[[VAL_13]] : f32 to i64
// CHECK:             %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : i64 to index
// CHECK:             %[[VAL_16:.*]] = arith.subf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_16]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_2]] : index
// CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_15]] : index
// CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_1]] : index
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_1]], %[[VAL_19]] : index
// CHECK:             %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_21]], %[[VAL_2]] : index
// CHECK:             %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_2]], %[[VAL_21]] : index
// CHECK:             %[[VAL_24:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : index to i64
// CHECK:             %[[VAL_26:.*]] = arith.sitofp %[[VAL_25]] : i64 to f32
// CHECK:             %[[VAL_27:.*]] = arith.addf %[[VAL_26]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_28:.*]] = arith.divf %[[VAL_27]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_29:.*]] = arith.subf %[[VAL_28]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_30:.*]] = math.floor %[[VAL_29]] : f32
// CHECK:             %[[VAL_31:.*]] = arith.fptosi %[[VAL_30]] : f32 to i64
// CHECK:             %[[VAL_32:.*]] = arith.index_cast %[[VAL_31]] : i64 to index
// CHECK:             %[[VAL_33:.*]] = arith.subf %[[VAL_29]], %[[VAL_30]] : f32
// CHECK:             %[[VAL_34:.*]] = arith.cmpf ogt, %[[VAL_33]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_32]], %[[VAL_2]] : index
// CHECK:             %[[VAL_36:.*]] = arith.select %[[VAL_34]], %[[VAL_35]], %[[VAL_32]] : index
// CHECK:             %[[VAL_37:.*]] = arith.cmpi slt, %[[VAL_36]], %[[VAL_1]] : index
// CHECK:             %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_1]], %[[VAL_36]] : index
// CHECK:             %[[VAL_39:.*]] = arith.cmpi sgt, %[[VAL_38]], %[[VAL_2]] : index
// CHECK:             %[[VAL_40:.*]] = arith.select %[[VAL_39]], %[[VAL_2]], %[[VAL_38]] : index
// CHECK:             %[[VAL_41:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_23]], %[[VAL_40]]] : tensor<1x1x2x2xi8>
// CHECK:             linalg.yield %[[VAL_41]] : i8
// CHECK:           } -> tensor<1x1x4x4xi8>
// CHECK:           return %[[VAL_5]] : tensor<1x1x4x4xi8>
// CHECK:         }
func.func @nearest_round_prefer_floor(%arg0: tensor<1x1x2x2xi8>) -> tensor<1x1x4x4xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %resize = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x2xi8>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xi8>
  return %resize : tensor<1x1x4x4xi8>
}
// -----
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:   func.func @linear_half_pixel_sizes(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = tensor.empty() : tensor<1x1x4x4xf32>
// CHECK:           %[[VAL_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_6]] : tensor<1x1x4x4xf32>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_11:.*]] = arith.sitofp %[[VAL_10]] : i64 to f32
// CHECK:             %[[VAL_12:.*]] = arith.addf %[[VAL_11]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_13:.*]] = arith.divf %[[VAL_12]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_14:.*]] = arith.subf %[[VAL_13]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_15:.*]] = math.floor %[[VAL_14]] : f32
// CHECK:             %[[VAL_16:.*]] = arith.fptosi %[[VAL_15]] : f32 to i64
// CHECK:             %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : i64 to index
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_4]] : index
// CHECK:             %[[VAL_19:.*]] = arith.subf %[[VAL_14]], %[[VAL_15]] : f32
// CHECK:             %[[VAL_20:.*]] = arith.subf %[[VAL_5]], %[[VAL_19]] : f32
// CHECK:             %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_17]], %[[VAL_1]] : index
// CHECK:             %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_1]], %[[VAL_17]] : index
// CHECK:             %[[VAL_23:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_4]] : index
// CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_4]], %[[VAL_22]] : index
// CHECK:             %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_1]] : index
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_1]], %[[VAL_18]] : index
// CHECK:             %[[VAL_27:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_4]] : index
// CHECK:             %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_4]], %[[VAL_26]] : index
// CHECK:             %[[VAL_29:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : index to i64
// CHECK:             %[[VAL_31:.*]] = arith.sitofp %[[VAL_30]] : i64 to f32
// CHECK:             %[[VAL_32:.*]] = arith.addf %[[VAL_31]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_33:.*]] = arith.divf %[[VAL_32]], %[[VAL_3]] : f32
// CHECK:             %[[VAL_34:.*]] = arith.subf %[[VAL_33]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_35:.*]] = math.floor %[[VAL_34]] : f32
// CHECK:             %[[VAL_36:.*]] = arith.fptosi %[[VAL_35]] : f32 to i64
// CHECK:             %[[VAL_37:.*]] = arith.index_cast %[[VAL_36]] : i64 to index
// CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_4]] : index
// CHECK:             %[[VAL_39:.*]] = arith.subf %[[VAL_34]], %[[VAL_35]] : f32
// CHECK:             %[[VAL_40:.*]] = arith.subf %[[VAL_5]], %[[VAL_39]] : f32
// CHECK:             %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_1]] : index
// CHECK:             %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[VAL_1]], %[[VAL_37]] : index
// CHECK:             %[[VAL_43:.*]] = arith.cmpi sgt, %[[VAL_42]], %[[VAL_4]] : index
// CHECK:             %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[VAL_4]], %[[VAL_42]] : index
// CHECK:             %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_38]], %[[VAL_1]] : index
// CHECK:             %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[VAL_1]], %[[VAL_38]] : index
// CHECK:             %[[VAL_47:.*]] = arith.cmpi sgt, %[[VAL_46]], %[[VAL_4]] : index
// CHECK:             %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[VAL_4]], %[[VAL_46]] : index
// CHECK:             %[[VAL_49:.*]] = arith.mulf %[[VAL_20]], %[[VAL_40]] : f32
// CHECK:             %[[VAL_50:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_24]], %[[VAL_44]]] : tensor<1x1x2x2xf32>
// CHECK:             %[[VAL_51:.*]] = arith.mulf %[[VAL_50]], %[[VAL_49]] : f32
// CHECK:             %[[VAL_52:.*]] = arith.addf %[[VAL_51]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_53:.*]] = arith.mulf %[[VAL_19]], %[[VAL_40]] : f32
// CHECK:             %[[VAL_54:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_28]], %[[VAL_44]]] : tensor<1x1x2x2xf32>
// CHECK:             %[[VAL_55:.*]] = arith.mulf %[[VAL_54]], %[[VAL_53]] : f32
// CHECK:             %[[VAL_56:.*]] = arith.addf %[[VAL_52]], %[[VAL_55]] : f32
// CHECK:             %[[VAL_57:.*]] = arith.mulf %[[VAL_20]], %[[VAL_39]] : f32
// CHECK:             %[[VAL_58:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_24]], %[[VAL_48]]] : tensor<1x1x2x2xf32>
// CHECK:             %[[VAL_59:.*]] = arith.mulf %[[VAL_58]], %[[VAL_57]] : f32
// CHECK:             %[[VAL_60:.*]] = arith.addf %[[VAL_56]], %[[VAL_59]] : f32
// CHECK:             %[[VAL_61:.*]] = arith.mulf %[[VAL_19]], %[[VAL_39]] : f32
// CHECK:             %[[VAL_62:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_28]], %[[VAL_48]]] : tensor<1x1x2x2xf32>
// CHECK:             %[[VAL_63:.*]] = arith.mulf %[[VAL_62]], %[[VAL_61]] : f32
// CHECK:             %[[VAL_64:.*]] = arith.addf %[[VAL_60]], %[[VAL_63]] : f32
// CHECK:             linalg.yield %[[VAL_64]] : f32
// CHECK:           } -> tensor<1x1x4x4xf32>
// CHECK:           return %[[VAL_7]] : tensor<1x1x4x4xf32>
// CHECK:         }
func.func @linear_half_pixel_sizes(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[1, 1, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
  %resize = "onnx.Resize"(%arg0, %none, %none, %sizes) {coordinate_transformation_mode = "half_pixel", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x2x2xf32>, none, none, tensor<4xi64>) -> tensor<1x1x4x4xf32>
  return %resize : tensor<1x1x4x4xf32>
}
// -----
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL:   func.func @linear_align_corners_bf16(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = tensor.empty() : tensor<1x1x4x4xbf16>
// CHECK:           %[[VAL_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_5]] : tensor<1x1x4x4xbf16>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: bf16):
// CHECK:             %[[VAL_8:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : index to i64
// CHECK:             %[[VAL_10:.*]] = arith.sitofp %[[VAL_9]] : i64 to f32
// CHECK:             %[[VAL_11:.*]] = arith.divf %[[VAL_10]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_12:.*]] = math.floor %[[VAL_11]] : f32
// CHECK:             %[[VAL_13:.*]] = arith.fptosi %[[VAL_12]] : f32 to i64
// CHECK:             %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i64 to index
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_3]] : index
// CHECK:             %[[VAL_16:.*]] = arith.subf %[[VAL_11]], %[[VAL_12]] : f32
// CHECK:             %[[VAL_17:.*]] = arith.subf %[[VAL_4]], %[[VAL_16]] : f32
// CHECK:             %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_1]], %[[VAL_14]] : index
// CHECK:             %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_3]] : index
// CHECK:             %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_3]], %[[VAL_19]] : index
// CHECK:             %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_1]] : index
// CHECK:             %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_1]], %[[VAL_15]] : index
// CHECK:             %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_3]] : index
// CHECK:             %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_3]], %[[VAL_23]] : index
// CHECK:             %[[VAL_26:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : index to i64
// CHECK:             %[[VAL_28:.*]] = arith.sitofp %[[VAL_27]] : i64 to f32
// CHECK:             %[[VAL_29:.*]] = arith.divf %[[VAL_28]], %[[VAL_2]] : f32
// CHECK:             %[[VAL_30:.*]] = math.floor %[[VAL_29]] : f32
// CHECK:             %[[VAL_31:.*]] = arith.fptosi %[[VAL_30]] : f32 to i64
// CHECK:             %[[VAL_32:.*]] = arith.index_cast %[[VAL_31]] : i64 to index
// CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_3]] : index
// CHECK:             %[[VAL_34:.*]] = arith.subf %[[VAL_29]], %[[VAL_30]] : f32
// CHECK:             %[[VAL_35:.*]] = arith.subf %[[VAL_4]], %[[VAL_34]] : f32
// CHECK:             %[[VAL_36:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_1]] : index
// CHECK:             %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_1]], %[[VAL_32]] : index
// CHECK:             %[[VAL_38:.*]] = arith.cmpi sgt, %[[VAL_37]], %[[VAL_3]] : index
// CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_38]], %[[VAL_3]], %[[VAL_37]] : index
// CHECK:             %[[VAL_40:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_1]] : index
// CHECK:             %[[VAL_41:.*]] = arith.select %[[VAL_40]], %[[VAL_1]], %[[VAL_33]] : index
// CHECK:             %[[VAL_42:.*]] = arith.cmpi sgt, %[[VAL_41]], %[[VAL_3]] : index
// CHECK:             %[[VAL_43:.*]] = arith.select %[[VAL_42]], %[[VAL_3]], %[[VAL_41]] : index
// CHECK:             %[[VAL_44:.*]] = arith.mulf %[[VAL_17]], %[[VAL_35]] : f32
// CHECK:             %[[VAL_45:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_21]], %[[VAL_39]]] : tensor<1x1x2x2xbf16>
// CHECK:             %[[VAL_46:.*]] = arith.extf %[[VAL_45]] : bf16 to f32
// CHECK:             %[[VAL_47:.*]] = arith.mulf %[[VAL_46]], %[[VAL_44]] : f32
// CHECK:             %[[VAL_48:.*]] = arith.addf %[[VAL_47]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_49:.*]] = arith.mulf %[[VAL_16]], %[[VAL_35]] : f32
// CHECK:             %[[VAL_50:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_25]], %[[VAL_39]]] : tensor<1x1x2x2xbf16>
// CHECK:             %[[VAL_51:.*]] = arith.extf %[[VAL_50]] : bf16 to f32
// CHECK:             %[[VAL_52:.*]] = arith.mulf %[[VAL_51]], %[[VAL_49]] : f32
// CHECK:             %[[VAL_53:.*]] = arith.addf %[[VAL_48]], %[[VAL_52]] : f32
// CHECK:             %[[VAL_54:.*]] = arith.mulf %[[VAL_17]], %[[VAL_34]] : f32
// CHECK:             %[[VAL_55:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_21]], %[[VAL_43]]] : tensor<1x1x2x2xbf16>
// CHECK:             %[[VAL_56:.*]] = arith.extf %[[VAL_55]] : bf16 to f32
// CHECK:             %[[VAL_57:.*]] = arith.mulf %[[VAL_56]], %[[VAL_54]] : f32
// CHECK:             %[[VAL_58:.*]] = arith.addf %[[VAL_53]], %[[VAL_57]] : f32
// CHECK:             %[[VAL_59:.*]] = arith.mulf %[[VAL_16]], %[[VAL_34]] : f32
// CHECK:             %[[VAL_60:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_1]], %[[VAL_25]], %[[VAL_43]]] : tensor<1x1x2x2xbf16>
// CHECK:             %[[VAL_61:.*]] = arith.extf %[[VAL_60]] : bf16 to f32
// CHECK:             %[[VAL_62:.*]] = arith.mulf %[[VAL_61]], %[[VAL_59]] : f32
// CHECK:             %[[VAL_63:.*]] = arith.addf %[[VAL_58]], %[[VAL_62]] : f32
// CHECK:             %[[VAL_64:.*]] = arith.truncf %[[VAL_63]] : f32 to bf16
// CHECK:             linalg.yield %[[VAL_64]] : bf16
// CHECK:           } -> tensor<1x1x4x4xbf16>
// CHECK:           return %[[VAL_6]] : tensor<1x1x4x4xbf16>
// CHECK:         }
func.func @linear_align_corners_bf16(%arg0: tensor<1x1x2x2xbf16>) -> tensor<1x1x4x4xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %resize = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "align_corners", mode = "linear", nearest_mode = "floor"} : (tensor<1x1x2x2xbf16>, none, tensor<4xf32>, none) -> tensor<1x1x4x4xbf16>
  return %resize : tensor<1x1x4x4xbf16>
}
// -----
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL:   func.func @nearest_5d(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x2x2x2x2xf32>) -> tensor<1x2x4x4x4xf32> {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x2x4x4x4xf32>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_3]] : tensor<1x2x4x4x4xf32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32):
// CHECK:             %[[VAL_6:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i64
// CHECK:             %[[VAL_8:.*]] = arith.sitofp %[[VAL_7]] : i64 to f32
// CHECK:             %[[VAL_9:.*]] = math.floor %[[VAL_8]] : f32
// CHECK:             %[[VAL_10:.*]] = arith.fptosi %[[VAL_9]] : f32 to i64
// CHECK:             %[[VAL_11:.*]] = arith.index_cast %[[VAL_10]] : i64 to index
// CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_2]] : index
// CHECK:             %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_2]], %[[VAL_11]] : index
// CHECK:             %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_1]] : index
// CHECK:             %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_1]], %[[VAL_13]] : index
// CHECK:             %[[VAL_16:.*]] = linalg.index 2 : index
// CHECK:             %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : index to i64
// CHECK:             %[[VAL_18:.*]] = arith.sitofp %[[VAL_17]] : i64 to f32
// CHECK:             %[[VAL_19:.*]] = arith.divf %[[VAL_18]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_20:.*]] = math.floor %[[VAL_19]] : f32
// CHECK:             %[[VAL_21:.*]] = arith.fptosi %[[VAL_20]] : f32 to i64
// CHECK:             %[[VAL_22:.*]] = arith.index_cast %[[VAL_21]] : i64 to index
// CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_2]] : index
// CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_2]], %[[VAL_22]] : index
// CHECK:             %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_1]] : index
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_1]], %[[VAL_24]] : index
// CHECK:             %[[VAL_27:.*]] = linalg.index 3 : index
// CHECK:             %[[VAL_28:.*]] = arith.index_cast %[[VAL_27]] : index to i64
// CHECK:             %[[VAL_29:.*]] = arith.sitofp %[[VAL_28]] : i64 to f32
// CHECK:             %[[VAL_30:.*]] = arith.divf %[[VAL_29]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_31:.*]] = math.floor %[[VAL_30]] : f32
// CHECK:             %[[VAL_32:.*]] = arith.fptosi %[[VAL_31]] : f32 to i64
// CHECK:             %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : i64 to index
// CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_2]] : index
// CHECK:             %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_2]], %[[VAL_33]] : index
// CHECK:             %[[VAL_36:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_1]] : index
// CHECK:             %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_1]], %[[VAL_35]] : index
// CHECK:             %[[VAL_38:.*]] = linalg.index 4 : index
// CHECK:             %[[VAL_39:.*]] = arith.index_cast %[[VAL_38]] : index to i64
// CHECK:             %[[VAL_40:.*]] = arith.sitofp %[[VAL_39]] : i64 to f32
// CHECK:             %[[VAL_41:.*]] = arith.divf %[[VAL_40]], %[[VAL_0]] : f32
// CHECK:             %[[VAL_42:.*]] = math.floor %[[VAL_41]] : f32
// CHECK:             %[[VAL_43:.*]] = arith.fptosi %[[VAL_42]] : f32 to i64
// CHECK:             %[[VAL_44:.*]] = arith.index_cast %[[VAL_43]] : i64 to index
// CHECK:             %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_44]], %[[VAL_2]] : index
// CHECK:             %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[VAL_2]], %[[VAL_44]] : index
// CHECK:             %[[VAL_47:.*]] = arith.cmpi sgt, %[[VAL_46]], %[[VAL_1]] : index
// CHECK:             %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[VAL_1]], %[[VAL_46]] : index
// CHECK:             %[[VAL_49:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[VAL_2]], %[[VAL_15]], %[[VAL_26]], %[[VAL_37]], %[[VAL_48]]] : tensor<1x2x2x2x2xf32>
// CHECK:             linalg.yield %[[VAL_49]] : f32
// CHECK:           } -> tensor<1x2x4x4x4xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x2x4x4x4xf32>
// CHECK:         }
func.func @nearest_5d(%arg0: tensor<1x2x2x2x2xf32>) -> tensor<1x2x4x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0, 2.0]> : tensor<5xf32>} : () -> tensor<5xf32>
  %resize = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<1x2x2x2x2xf32>, none, tensor<5xf32>, none) -> tensor<1x2x4x4x4xf32>
  return %resize : tensor<1x2x4x4x4xf32>
}
// -----
// CHECK-LABEL:   func.func @unsupported_rank_is_left_for_fallback(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x2x2xf32>) -> tensor<1x4x4xf32> {
// CHECK:           %[[VAL_0:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           %[[VAL_1:.*]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<3xf32>
// CHECK:           %[[VAL_2:.*]] = "onnx.Resize"(%[[ARG0]], %[[VAL_0]], %[[VAL_1]], %[[VAL_0]]) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<1x2x2xf32>, none, tensor<3xf32>, none) -> tensor<1x4x4xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x4x4xf32>
// CHECK:         }
func.func @unsupported_rank_is_left_for_fallback(%arg0: tensor<1x2x2xf32>) -> tensor<1x4x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %resize = "onnx.Resize"(%arg0, %none, %scales, %none) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<1x2x2xf32>, none, tensor<3xf32>, none) -> tensor<1x4x4xf32>
  return %resize : tensor<1x4x4xf32>
}
