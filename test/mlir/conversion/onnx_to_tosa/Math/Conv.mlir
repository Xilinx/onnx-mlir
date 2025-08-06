// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa=grouped-conv-threshold=4 -cse %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x15x15xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x15x15xf32>
  return %0 : tensor<5x2x15x15xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_stride_13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x15x15xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[5, 245, 245, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_3_]], [[VAR_4_]] : (tensor<5x256x256x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x245x245x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.conv2d [[VAR_5_]], [[VAR_2_]], [[PARAM_2_]], [[VAR_6_]], [[VAR_6_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x245x245x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x15x15x2xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_9_:%.+]] = tosa.transpose [[VAR_7_]], [[VAR_8_]] : (tensor<5x15x15x2xf32>, tensor<4xi32>) -> tensor<5x2x15x15xf32>
// CHECK:           return [[VAR_9_]] : tensor<5x2x15x15xf32>
// CHECK:         }
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>) ->  tensor<5x2x197x199xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x197x199xf32>
  return %0 : tensor<5x2x197x199xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_novalue
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>) -> tensor<5x2x197x199xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]], [[VAR_4_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 3, 2, 4>, stride = array<i64: 1, 1>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x197x199x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_7_:%.+]] = tosa.transpose [[VAR_5_]], [[VAR_6_]] : (tensor<5x197x199x2xf32>, tensor<4xi32>) -> tensor<5x2x197x199xf32>
// CHECK:           return [[VAR_7_]] : tensor<5x2x197x199xf32>
// CHECK:         }
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<7x3x64x64xf32>) ->   tensor<5x7x15x15xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<7x3x64x64xf32>, none) ->  tensor<5x7x15x15xf32>
  return %0 :  tensor<5x7x15x15xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_no_dilation_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<7x3x64x64xf32>) -> tensor<5x7x15x15xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<7x3x64x64xf32>, tensor<4xi32>) -> tensor<7x64x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<7xf32>}> : () -> tensor<7xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<[5, 246, 246, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_4_]], [[VAR_5_]] : (tensor<5x256x256x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x246x246x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.conv2d [[VAR_6_]], [[VAR_2_]], [[VAR_3_]], [[VAR_7_]], [[VAR_7_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 13, 13>} : (tensor<5x246x246x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x15x15x7xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_10_:%.+]] = tosa.transpose [[VAR_8_]], [[VAR_9_]] : (tensor<5x15x15x7xf32>, tensor<4xi32>) -> tensor<5x7x15x15xf32>
// CHECK:           return [[VAR_10_]] : tensor<5x7x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_bf16_no_bias(%arg0: tensor<5x3x256x256xbf16>, %arg1 : tensor<7x3x64x64xbf16>) ->   tensor<5x7x15x15xbf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [13, 13]} : (tensor<5x3x256x256xbf16>, tensor<7x3x64x64xbf16>, none) ->  tensor<5x7x15x15xbf16>
  return %0 :  tensor<5x7x15x15xbf16>
// CHECK-LABEL:  func.func @test_onnx_conv2d_bf16_no_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xbf16>, [[PARAM_1_:%.+]]: tensor<7x3x64x64xbf16>) -> tensor<5x7x15x15xbf16> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xbf16>, tensor<4xi32>) -> tensor<5x256x256x3xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<7x3x64x64xbf16>, tensor<4xi32>) -> tensor<7x64x64x3xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<7xbf16>}> : () -> tensor<7xbf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<[5, 246, 246, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_4_]], [[VAR_5_]] : (tensor<5x256x256x3xbf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x246x246x3xbf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.conv2d [[VAR_6_]], [[VAR_2_]], [[VAR_3_]], [[VAR_7_]], [[VAR_7_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 13, 13>} : (tensor<5x246x246x3xbf16>, tensor<7x64x64x3xbf16>, tensor<7xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<5x15x15x7xbf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_10_:%.+]] = tosa.transpose [[VAR_8_]], [[VAR_9_]] : (tensor<5x15x15x7xbf16>, tensor<4xi32>) -> tensor<5x7x15x15xbf16>
// CHECK:           return [[VAR_10_]] : tensor<5x7x15x15xbf16>
}

// -----
func.func @test_onnx_conv2d_f16_no_bias(%arg0: tensor<5x3x256x256xf16>, %arg1 : tensor<7x3x64x64xf16>) ->   tensor<5x7x15x15xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [13, 13]} : (tensor<5x3x256x256xf16>, tensor<7x3x64x64xf16>, none) ->  tensor<5x7x15x15xf16>
  return %0 :  tensor<5x7x15x15xf16>
// CHECK-LABEL:  func.func @test_onnx_conv2d_f16_no_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf16>, [[PARAM_1_:%.+]]: tensor<7x3x64x64xf16>) -> tensor<5x7x15x15xf16> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf16>, tensor<4xi32>) -> tensor<5x256x256x3xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<7x3x64x64xf16>, tensor<4xi32>) -> tensor<7x64x64x3xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<7xf16>}> : () -> tensor<7xf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<[5, 246, 246, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_4_]], [[VAR_5_]] : (tensor<5x256x256x3xf16>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x246x246x3xf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.conv2d [[VAR_6_]], [[VAR_2_]], [[VAR_3_]], [[VAR_7_]], [[VAR_7_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 13, 13>} : (tensor<5x246x246x3xf16>, tensor<7x64x64x3xf16>, tensor<7xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x15x15x7xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_10_:%.+]] = tosa.transpose [[VAR_8_]], [[VAR_9_]] : (tensor<5x15x15x7xf16>, tensor<4xi32>) -> tensor<5x7x15x15xf16>
// CHECK:           return [[VAR_10_]] : tensor<5x7x15x15xf16>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x256x260xf32>, %arg1 : tensor<2x3x60x64xf32>) ->  tensor<5x2x197x197xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) : (tensor<5x3x256x260xf32>, tensor<2x3x60x64xf32>, none) ->  tensor<5x2x197x197xf32>
  return %0 : tensor<5x2x197x197xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_no_dilation_pad_stride
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x260xf32>, [[PARAM_1_:%.+]]: tensor<2x3x60x64xf32>) -> tensor<5x2x197x197xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x260xf32>, tensor<4xi32>) -> tensor<5x256x260x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x60x64xf32>, tensor<4xi32>) -> tensor<2x60x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]], [[VAR_4_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x256x260x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x197x197x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_7_:%.+]] = tosa.transpose [[VAR_5_]], [[VAR_6_]] : (tensor<5x197x197x2xf32>, tensor<4xi32>) -> tensor<5x2x197x197xf32>
// CHECK:           return [[VAR_7_]] : tensor<5x2x197x197xf32>
// CHECK:         }
}

// -----
func.func @test_onnx_conv2d_group(%arg0: tensor<5x64x256x256xf32>, %arg1 : tensor<12x16x45x45xf32>, %arg2: tensor<12xf32>) ->  tensor<5x12x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 1, 1, 1], strides = [13, 13], group = 4 : si64} : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) ->  tensor<5x12x17x17xf32>
  return %0 : tensor<5x12x17x17xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x64x256x256xf32>, [[PARAM_1_:%.+]]: tensor<12x16x45x45xf32>, [[PARAM_2_:%.+]]: tensor<12xf32>) -> tensor<5x12x17x17xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x64x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x64xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<12x16x45x45xf32>, tensor<4xi32>) -> tensor<12x45x45x16xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[5, 252, 252, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[VAR_1_]], [[VAR_3_]], [[VAR_4_]] : (tensor<5x256x256x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x252x252x64xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[5, 252, 252, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.slice [[VAR_5_]], [[VAR_3_]], [[VAR_6_]] : (tensor<5x252x252x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[3, 45, 45, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_3_]], [[VAR_8_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_10_]], [[VAR_11_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.conv2d [[VAR_7_]], [[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_13_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = tosa.const_shape  {value = dense<[0, 0, 0, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_16_:%.+]] = tosa.slice [[VAR_5_]], [[VAR_15_]], [[VAR_6_]] : (tensor<5x252x252x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = tosa.const_shape  {value = dense<[3, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_18_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_17_]], [[VAR_8_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_11_]], [[VAR_11_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = tosa.conv2d [[VAR_16_]], [[VAR_18_]], [[VAR_19_]], [[VAR_13_]], [[VAR_13_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = tosa.const_shape  {value = dense<[0, 0, 0, 32]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_22_:%.+]] = tosa.slice [[VAR_5_]], [[VAR_21_]], [[VAR_6_]] : (tensor<5x252x252x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = tosa.const_shape  {value = dense<[6, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_24_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_2_]]3, [[VAR_8_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = tosa.const_shape  {value = dense<6> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_26_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_25_]], [[VAR_11_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = tosa.conv2d [[VAR_22_]], [[VAR_24_]], [[VAR_26_]], [[VAR_13_]], [[VAR_13_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = tosa.const_shape  {value = dense<[0, 0, 0, 48]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_29_:%.+]] = tosa.slice [[VAR_5_]], [[VAR_28_]], [[VAR_6_]] : (tensor<5x252x252x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<5x252x252x16xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = tosa.const_shape  {value = dense<[9, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_31_:%.+]] = tosa.slice [[VAR_2_]], [[VAR_30_]], [[VAR_8_]] : (tensor<12x45x45x16xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = tosa.const_shape  {value = dense<9> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_33_:%.+]] = tosa.slice [[PARAM_2_]], [[VAR_32_]], [[VAR_11_]] : (tensor<12xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK:           [[VAR_34_:%.+]] = tosa.conv2d [[VAR_29_]], [[VAR_31_]], [[VAR_33_]], [[VAR_13_]], [[VAR_13_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 13, 13>} : (tensor<5x252x252x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = tosa.concat [[VAR_14_]], [[VAR_20_]], [[VAR_27_]], [[VAR_34_]] {axis = 3 : i32} : (tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>) -> tensor<5x17x17x12xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_37_:%.+]] = tosa.transpose [[VAR_35_]], [[VAR_36_]] : (tensor<5x17x17x12xf32>, tensor<4xi32>) -> tensor<5x12x17x17xf32>
// CHECK:           return [[VAR_37_]] : tensor<5x12x17x17xf32>
}

// -----
func.func @test_onnx_conv2d_autopad(%arg0: tensor<5x3x125x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x125x256xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_LOWER"} : (tensor<5x3x125x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x125x256xf32>
  return %0 : tensor<5x2x125x256xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_autopad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x125x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x125x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x125x256xf32>, tensor<4xi32>) -> tensor<5x125x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[PARAM_2_]], [[VAR_3_]], [[VAR_3_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 32, 31, 32, 31>, stride = array<i64: 1, 1>} : (tensor<5x125x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<5x125x256x2xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_6_:%.+]] = tosa.transpose [[VAR_4_]], [[VAR_5_]] : (tensor<5x125x256x2xf32>, tensor<4xi32>) -> tensor<5x2x125x256xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x2x125x256xf32>
// CHECK:         }
}

// -----
func.func @test_onnx_conv2d_group_higher_4(%arg0: tensor<5x128x256x256xf32>, %arg1 : tensor<16x16x45x45xf32>, %arg2: tensor<16xf32>) ->  tensor<5x16x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 8 : si64, pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x128x256x256xf32>, tensor<16x16x45x45xf32>, tensor<16xf32>) ->  tensor<5x16x17x17xf32>
  return %0 : tensor<5x16x17x17xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group_higher_4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x128x256x256xf32>, [[PARAM_1_:%.+]]: tensor<16x16x45x45xf32>, [[PARAM_2_:%.+]]: tensor<16xf32>) -> tensor<5x16x17x17xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {auto_pad = "NOTSET", group = 8 : si64, pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x128x256x256xf32>, tensor<16x16x45x45xf32>, tensor<16xf32>) -> tensor<5x16x17x17xf32>
// CHECK:           return [[VAR_0_]] : tensor<5x16x17x17xf32>
}

// -----
func.func @test_onnx_conv2d_group_to_depthwise(%arg0: tensor<32x48x112x112xf32>, %arg1 : tensor<48x1x3x3xf32>, %arg2: tensor<48xf32>) ->  tensor<32x48x112x112xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 48 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1395", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<32x48x112x112xf32>, tensor<48x1x3x3xf32>, tensor<48xf32>) -> tensor<32x48x112x112xf32>
  return %0 : tensor<32x48x112x112xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group_to_depthwise
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<32x48x112x112xf32>, [[PARAM_1_:%.+]]: tensor<48x1x3x3xf32>, [[PARAM_2_:%.+]]: tensor<48xf32>) -> tensor<32x48x112x112xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<32x48x112x112xf32>, tensor<4xi32>) -> tensor<32x112x112x48xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_2_]] : (tensor<48x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x48x1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[3, 3, 48, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_3_]], [[VAR_4_]] : (tensor<3x3x48x1xf32>, !tosa.shape<4>) -> tensor<3x3x48x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.depthwise_conv2d [[VAR_1_]], [[VAR_5_]], [[PARAM_2_]], [[VAR_6_]], [[VAR_6_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<32x112x112x48xf32>, tensor<3x3x48x1xf32>, tensor<48xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32x112x112x48xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_9_:%.+]] = tosa.transpose [[VAR_7_]], [[VAR_8_]] : (tensor<32x112x112x48xf32>, tensor<4xi32>) -> tensor<32x48x112x112xf32>
// CHECK:           return [[VAR_9_]] : tensor<32x48x112x112xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple(%arg0: tensor<32x24x112x112xf32>, %arg1 : tensor<48x1x3x3xf32>, %arg2: tensor<48xf32>) ->  tensor<32x48x112x112xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 24 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_1395", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<32x24x112x112xf32>, tensor<48x1x3x3xf32>, tensor<48xf32>) -> tensor<32x48x112x112xf32>
  return %0 : tensor<32x48x112x112xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group_to_depthwise_integer_multiple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<32x24x112x112xf32>, [[PARAM_1_:%.+]]: tensor<48x1x3x3xf32>, [[PARAM_2_:%.+]]: tensor<48xf32>) -> tensor<32x48x112x112xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<32x24x112x112xf32>, tensor<4xi32>) -> tensor<32x112x112x24xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[2, 3, 0, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_2_]] : (tensor<48x1x3x3xf32>, tensor<4xi32>) -> tensor<3x3x48x1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[3, 3, 24, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_3_]], [[VAR_4_]] : (tensor<3x3x48x1xf32>, !tosa.shape<4>) -> tensor<3x3x24x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.depthwise_conv2d [[VAR_1_]], [[VAR_5_]], [[PARAM_2_]], [[VAR_6_]], [[VAR_6_]] {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<32x112x112x24xf32>, tensor<3x3x24x2xf32>, tensor<48xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32x112x112x48xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_9_:%.+]] = tosa.transpose [[VAR_7_]], [[VAR_8_]] : (tensor<32x112x112x48xf32>, tensor<4xi32>) -> tensor<32x48x112x112xf32>
// CHECK:           return [[VAR_9_]] : tensor<32x48x112x112xf32>
}

// -----

func.func @test_onnx_conv2d_dyn_shapes(%arg0: tensor<?x?x?x?xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<?x?x?x?xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<?x?x?x?xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_dyn_shapes
// CHECK: onnx.Conv
}

// -----

func.func @test_onnx_conv2d_dyn_shapes_no_rank(%arg0: tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2: tensor<*xf32>) ->  tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) ->  tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_dyn_shapes_no_rank
// CHECK: onnx.Conv
}

// -----

func.func @test_onnx_conv2d_dyn_shapes_with_shape_inference(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<?x?x?x?xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_dyn_shapes_with_shape_inference
// CHECK: tosa.conv
}