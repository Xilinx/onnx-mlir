// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_onnx_to_matmul2d(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_to_matmul2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<4x16xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<4x8xf32>, !tosa.shape<3>) -> tensor<1x4x8xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 8, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_2_]] : (tensor<8x16xf32>, !tosa.shape<3>) -> tensor<1x8x16xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_3_]] : (tensor<1x4x8xf32>, tensor<1x8x16xf32>) -> tensor<1x4x16xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<[4, 16]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_4_]], [[VAR_5_]] : (tensor<1x4x16xf32>, !tosa.shape<2>) -> tensor<4x16xf32>
// CHECK:           return [[VAR_6_]] : tensor<4x16xf32>
}

// -----

func.func @test_onnx_to_matmul3dbcast(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_to_matmul3dbcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 8, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_0_]] : (tensor<8x16xf32>, !tosa.shape<3>) -> tensor<1x8x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 400, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_2_]] : (tensor<100x4x8xf32>, !tosa.shape<3>) -> tensor<1x400x8xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[1, 0, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_4_]] : (tensor<1x8x16xf32>, tensor<3xi32>) -> tensor<8x1x16xf32>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_5_]], [[VAR_0_]] : (tensor<8x1x16xf32>, !tosa.shape<3>) -> tensor<1x8x16xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_6_]] : (tensor<1x400x8xf32>, tensor<1x8x16xf32>) -> tensor<1x400x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[100, 4, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<1x400x16xf32>, !tosa.shape<3>) -> tensor<100x4x16xf32>
// CHECK:           return [[VAR_9_]] : tensor<100x4x16xf32>
}

// -----

func.func @test_onnx_1d(%arg0 : tensor<6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6xf32>, [[PARAM_1_:%.+]]: tensor<6xf32>) -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<6xf32>, !tosa.shape<2>) -> tensor<1x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[6, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_2_]] : (tensor<6xf32>, !tosa.shape<2>) -> tensor<6x1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_4_]] : (tensor<1x6xf32>, !tosa.shape<3>) -> tensor<1x1x6xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[1, 6, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_3_]], [[VAR_6_]] : (tensor<6x1xf32>, !tosa.shape<3>) -> tensor<1x6x1xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.matmul [[VAR_5_]], [[VAR_7_]] : (tensor<1x1x6xf32>, tensor<1x6x1xf32>) -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<> : tensor<0xindex>} : () -> !tosa.shape<0>
// CHECK:           [[VAR_10_:%.+]] = tosa.reshape [[VAR_8_]], [[VAR_9_]] : (tensor<1x1x1xf32>, !tosa.shape<0>) -> tensor<f32>
// CHECK:           return [[VAR_10_]] : tensor<f32>
}

// -----

func.func @test_onnx_12d(%arg0 : tensor<6xf32>, %arg1 : tensor<6x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6x1xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_12d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<6xf32>, [[PARAM_1_:%.+]]: tensor<6x1xf32>) -> tensor<1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<6xf32>, !tosa.shape<2>) -> tensor<1x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_2_]] : (tensor<1x6xf32>, !tosa.shape<3>) -> tensor<1x1x6xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[1, 6, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_4_]] : (tensor<6x1xf32>, !tosa.shape<3>) -> tensor<1x6x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_5_]] : (tensor<1x1x6xf32>, tensor<1x6x1xf32>) -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_8_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_7_]] : (tensor<1x1x1xf32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK:           return [[VAR_8_]] : tensor<1xf32>
}

// -----

func.func @test_onnx_21d(%arg0 : tensor<2x6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_21d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x6xf32>, [[PARAM_1_:%.+]]: tensor<6xf32>) -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[6, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_0_]] : (tensor<6xf32>, !tosa.shape<2>) -> tensor<6x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[1, 2, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_2_]] : (tensor<2x6xf32>, !tosa.shape<3>) -> tensor<1x2x6xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[1, 6, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_4_]] : (tensor<6x1xf32>, !tosa.shape<3>) -> tensor<1x6x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_5_]] : (tensor<1x2x6xf32>, tensor<1x6x1xf32>) -> tensor<1x2x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_8_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_7_]] : (tensor<1x2x1xf32>, !tosa.shape<1>) -> tensor<2xf32>
// CHECK:           return [[VAR_8_]] : tensor<2xf32>
}

// -----

func.func @test_onnx_4d(%arg0 : tensor<10x10x6x2xf32>, %arg1 : tensor<10x10x2x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x10x6x2xf32>, tensor<10x10x2x6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_4d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10x6x2xf32>, [[PARAM_1_:%.+]]: tensor<10x10x2x6xf32>) -> tensor<10x10x6x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[100, 6, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<10x10x6x2xf32>, !tosa.shape<3>) -> tensor<100x6x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[100, 2, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_2_]] : (tensor<10x10x2x6xf32>, !tosa.shape<3>) -> tensor<100x2x6xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_3_]] : (tensor<100x6x2xf32>, tensor<100x2x6xf32>) -> tensor<100x6x6xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {value = dense<[10, 10, 6, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           [[VAR_6_:%.+]] = tosa.reshape [[VAR_4_]], [[VAR_5_]] : (tensor<100x6x6xf32>, !tosa.shape<4>) -> tensor<10x10x6x6xf32>
// CHECK:           return [[VAR_6_]] : tensor<10x10x6x6xf32>
}

// -----

func.func @test_onnx_4d_mixed(%arg0 : tensor<10x6x2xf32>, %arg1 : tensor<10x10x2x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x6x2xf32>, tensor<10x10x2x6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_4d_mixed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x6x2xf32>, [[PARAM_1_:%.+]]: tensor<10x10x2x6xf32>) -> tensor<10x10x6x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 10, 6, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<10x6x2xf32>, !tosa.shape<4>) -> tensor<1x10x6x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[1, 0, 2, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_2_]] : (tensor<1x10x6x2xf32>, tensor<4xi32>) -> tensor<10x1x6x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[10, 6, 2]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_3_]], [[VAR_4_]] : (tensor<10x1x6x2xf32>, !tosa.shape<3>) -> tensor<10x6x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[1, 2, 0, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_6_]] : (tensor<10x10x2x6xf32>, tensor<4xi32>) -> tensor<10x2x10x6xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[10, 2, 60]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<10x2x10x6xf32>, !tosa.shape<3>) -> tensor<10x2x60xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.matmul [[VAR_5_]], [[VAR_9_]] : (tensor<10x6x2xf32>, tensor<10x2x60xf32>) -> tensor<10x6x60xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[10, 6, 10, 6]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.reshape [[VAR_10_]], [[VAR_11_]] : (tensor<10x6x60xf32>, !tosa.shape<4>) -> tensor<10x6x10x6xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_14_:%.+]] = tosa.transpose [[VAR_12_]], [[VAR_13_]] : (tensor<10x6x10x6xf32>, tensor<4xi32>) -> tensor<10x10x6x6xf32>
// CHECK:           return [[VAR_14_]] : tensor<10x10x6x6xf32>
}

// -----

func.func @test_onnx_to_matmul4d_non_broadcastable(%arg0 : tensor<4x1x5x6xf32>, %arg1 : tensor<1x3x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x1x5x6xf32>, tensor<1x3x6x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_to_matmul4d_non_broadcastable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x1x5x6xf32>, [[PARAM_1_:%.+]]: tensor<1x3x6x7xf32>) -> tensor<4x3x5x7xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 20, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_0_]], [[VAR_0_]] : (tensor<4x1x5x6xf32>, !tosa.shape<3>) -> tensor<1x20x6xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_2_]] : (tensor<1x3x6x7xf32>, tensor<4xi32>) -> tensor<6x1x3x7xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[1, 6, 21]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[VAR_3_]], [[VAR_4_]] : (tensor<6x1x3x7xf32>, !tosa.shape<3>) -> tensor<1x6x21xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.matmul [[VAR_1_]], [[VAR_5_]] : (tensor<1x20x6xf32>, tensor<1x6x21xf32>) -> tensor<1x20x21xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<[4, 5, 3, 7]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.reshape [[VAR_6_]], [[VAR_7_]] : (tensor<1x20x21xf32>, !tosa.shape<4>) -> tensor<4x5x3x7xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_10_:%.+]] = tosa.transpose [[VAR_8_]], [[VAR_9_]] : (tensor<4x5x3x7xf32>, tensor<4xi32>) -> tensor<4x3x5x7xf32>
// CHECK:           return [[VAR_10_]] : tensor<4x3x5x7xf32>
}

// -----

func.func @test_onnx_to_matmul_7d_6d_broadcastable(%arg0: tensor<1x1x6x1x4x4xf32>, %arg1: tensor<4x2x6x2500x4x1xf32>) -> (tensor<*xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1x6x1x4x4xf32>, tensor<4x2x6x2500x4x1xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_to_matmul_7d_6d_broadcastable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x6x1x4x4xf32>, [[PARAM_1_:%.+]]: tensor<4x2x6x2500x4x1xf32>) -> tensor<4x2x6x2500x4x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1, 3, 4, 5]> : tensor<6xi32>}> : () -> tensor<6xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<1x1x6x1x4x4xf32>, tensor<6xi32>) -> tensor<6x1x1x1x4x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[6, 4, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_2_]] : (tensor<6x1x1x1x4x4xf32>, !tosa.shape<3>) -> tensor<6x4x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi32>}> : () -> tensor<6xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_4_]] : (tensor<4x2x6x2500x4x1xf32>, tensor<6xi32>) -> tensor<6x4x4x2x2500x1xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[6, 4, 20000]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_7_:%.+]] = tosa.reshape [[VAR_5_]], [[VAR_6_]] : (tensor<6x4x4x2x2500x1xf32>, !tosa.shape<3>) -> tensor<6x4x20000xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.matmul [[VAR_3_]], [[VAR_7_]] : (tensor<6x4x4xf32>, tensor<6x4x20000xf32>) -> tensor<6x4x20000xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.const_shape  {value = dense<[6, 4, 4, 2, 2500, 1]> : tensor<6xindex>} : () -> !tosa.shape<6>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.reshape [[VAR_8_]], [[VAR_9_]] : (tensor<6x4x20000xf32>, !tosa.shape<6>) -> tensor<6x4x4x2x2500x1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "tosa.const"() <{value = dense<[2, 3, 0, 4, 1, 5]> : tensor<6xi32>}> : () -> tensor<6xi32>
// CHECK:           [[VAR_12_:%.+]] = tosa.transpose [[VAR_10_]], [[VAR_11_]] : (tensor<6x4x4x2x2500x1xf32>, tensor<6xi32>) -> tensor<4x2x6x2500x4x1xf32>
// CHECK:           return [[VAR_12_]] : tensor<4x2x6x2500x4x1xf32>
}

// -----

func.func @test_onnx_to_matmul_8d_7d_broadcastable(%arg0: tensor<4x3x2x1x5x4x7x6xf32>, %arg1: tensor<2x9x1x1x6x8xf32>) -> (tensor<*xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x3x2x1x5x4x7x6xf32>, tensor<2x9x1x1x6x8xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_onnx_to_matmul_8d_7d_broadcastable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<4x3x2x1x5x4x7x6xf32>, [[PARAM_1_:%.+]]: tensor<2x9x1x1x6x8xf32>) -> tensor<4x3x2x9x5x4x7x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<[1, 1, 2, 9, 1, 1, 6, 8]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_0_]] : (tensor<2x9x1x1x6x8xf32>, !tosa.shape<8>) -> tensor<1x1x2x9x1x1x6x8xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<[2, 0, 1, 3, 4, 5, 6, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_2_]] : (tensor<4x3x2x1x5x4x7x6xf32>, tensor<8xi32>) -> tensor<2x4x3x1x5x4x7x6xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[2, 1680, 6]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.reshape [[VAR_3_]], [[VAR_4_]] : (tensor<2x4x3x1x5x4x7x6xf32>, !tosa.shape<3>) -> tensor<2x1680x6xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.const"() <{value = dense<[2, 6, 0, 1, 3, 4, 5, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.transpose [[VAR_1_]], [[VAR_6_]] : (tensor<1x1x2x9x1x1x6x8xf32>, tensor<8xi32>) -> tensor<2x6x1x1x9x1x1x8xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.const_shape  {value = dense<[2, 6, 72]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_9_:%.+]] = tosa.reshape [[VAR_7_]], [[VAR_8_]] : (tensor<2x6x1x1x9x1x1x8xf32>, !tosa.shape<3>) -> tensor<2x6x72xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.matmul [[VAR_5_]], [[VAR_9_]] : (tensor<2x1680x6xf32>, tensor<2x6x72xf32>) -> tensor<2x1680x72xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.const_shape  {value = dense<[2, 4, 3, 5, 4, 7, 9, 8]> : tensor<8xindex>} : () -> !tosa.shape<8>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.reshape [[VAR_10_]], [[VAR_11_]] : (tensor<2x1680x72xf32>, !tosa.shape<8>) -> tensor<2x4x3x5x4x7x9x8xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "tosa.const"() <{value = dense<[1, 2, 0, 6, 3, 4, 5, 7]> : tensor<8xi32>}> : () -> tensor<8xi32>
// CHECK:           [[VAR_14_:%.+]] = tosa.transpose [[VAR_12_]], [[VAR_13_]] : (tensor<2x4x3x5x4x7x9x8xf32>, tensor<8xi32>) -> tensor<4x3x2x9x5x4x7x8xf32>
// CHECK:           return [[VAR_14_]] : tensor<4x3x2x9x5x4x7x8xf32>
}

// -----
func.func @test_onnx_to_matmul3d_fp16(%arg0 : tensor<100x4x8xf16>, %arg1 : tensor<100x8x16xf16>) -> tensor<*xf16> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf16>, tensor<100x8x16xf16>) -> tensor<*xf16>
  "func.return"(%0) : (tensor<*xf16>) -> ()
  // CHECK:  %0 = tosa.matmul %arg0, %arg1 : (tensor<100x4x8xf16>, tensor<100x8x16xf16>) -> tensor<100x4x16xf32>
  // CHECK:  %1 = tosa.cast %0 : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16>
  // CHECK:  return %1 : tensor<100x4x16xf16>
}

// -----

func.func @test_onnx_to_matmul3d_bf16(%arg0 : tensor<100x4x8xbf16>, %arg1 : tensor<100x8x16xbf16>) -> tensor<*xbf16> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xbf16>, tensor<100x8x16xbf16>) -> tensor<*xbf16>
  "func.return"(%0) : (tensor<*xbf16>) -> ()
  // CHECK:   %0 = tosa.matmul %arg0, %arg1 : (tensor<100x4x8xbf16>, tensor<100x8x16xbf16>) -> tensor<100x4x16xf32>
  // CHECK:   %1 = tosa.cast %0 : (tensor<100x4x16xf32>) -> tensor<100x4x16xbf16>
  // CHECK:   return %1 : tensor<100x4x16xbf16>
}

// -----

func.func @test_onnx_to_matmul3d_fp32(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK:   %0 = tosa.matmul %arg0, %arg1 : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
  // CHECK:   return %0 : tensor<100x4x16xf32>
}

// -----

func.func @test_onnx_to_matmul2d_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-NOT: tosa.matmul
}

// -----

func.func @test_onnx_to_matmul3d_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-NOT: tosa.matmul
}
