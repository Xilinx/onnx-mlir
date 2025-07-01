// RUN: onnx-mlir-opt --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<8x32x64xf32>, tensor<8x32x64xf32>) {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<8x32x64xf32>, tensor<8x32x64xf32>)
    return %0, %1 : tensor<8x32x64xf32>, tensor<8x32x64xf32>
}

// CHECK-LABEL:  func.func @test_split_equal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> (tensor<8x32x64xf32>, tensor<8x32x64xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[8, 32, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<16x32x64xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x32x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[8, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_4_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_3_]], [[VAR_1_]] : (tensor<16x32x64xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x32x64xf32>
// CHECK:           return [[VAR_2_]], [[VAR_4_]] : tensor<8x32x64xf32>, tensor<8x32x64xf32>

// -----

func.func @test_split_variable(%arg0 : tensor<16x32x64xf16>) -> (tensor<16x2x64xf16>, tensor<16x30x64xf16>) {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xf16>, tensor<2xi64>) -> (tensor<16x2x64xf16>, tensor<16x30x64xf16>)
  "func.return"(%0, %1) : (tensor<16x2x64xf16>, tensor<16x30x64xf16>) -> ()
}

// CHECK-LABEL:  func.func @test_split_variable
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf16>) -> (tensor<16x2x64xf16>, tensor<16x30x64xf16>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[16, 2, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<16x32x64xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x2x64xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[0, 2, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[16, 30, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_5_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_3_]], [[VAR_4_]] : (tensor<16x32x64xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x30x64xf16>
// CHECK:           return [[VAR_2_]], [[VAR_5_]] : tensor<16x2x64xf16>, tensor<16x30x64xf16>

// -----

func.func @test_split_multiple(%arg0 : tensor<16x32x64xf16>) -> (tensor<16x4x64xf16>, tensor<16x8x64xf16>, tensor<16x20x64xf16>) {
  %split = "onnx.Constant"() {value = dense<[4, 8, 20]> : tensor<3xi64>} : () -> tensor<3xi64>
  %0, %1, %2 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xf16>, tensor<3xi64>) -> (tensor<16x4x64xf16>, tensor<16x8x64xf16>, tensor<16x20x64xf16>)
  "func.return"(%0, %1, %2) : (tensor<16x4x64xf16>, tensor<16x8x64xf16>, tensor<16x20x64xf16>) -> ()
}

// CHECK-LABEL:  func.func @test_split_multiple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf16>) -> (tensor<16x4x64xf16>, tensor<16x8x64xf16>, tensor<16x20x64xf16>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[16, 4, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<16x32x64xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x4x64xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[0, 4, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.const_shape  {value = dense<[16, 8, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_3_]], [[VAR_4_]] : (tensor<16x32x64xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x8x64xf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.const_shape  {value = dense<[0, 12, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.const_shape  {value = dense<[16, 20, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_8_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_6_]], [[VAR_7_]] : (tensor<16x32x64xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x20x64xf16>
// CHECK:           return [[VAR_2_]], [[VAR_5_]], [[VAR_8_]] : tensor<16x4x64xf16>, tensor<16x8x64xf16>, tensor<16x20x64xf16>


// -----

func.func @test_no_split(%arg0 : tensor<16x32x64xi32>) -> tensor<16x16x64xi32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64} : (tensor<16x32x64xi32>, none) -> (tensor<16x16x64xi32>, tensor<16x16x64xi32>)
  "func.return"(%0) : (tensor<16x16x64xi32>) -> ()
}

// CHECK-LABEL:  func.func @test_no_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xi32>) -> tensor<16x16x64xi32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[16, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<16x32x64xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x16x64xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[0, 16, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           return [[VAR_2_]] : tensor<16x16x64xi32>


// -----

func.func @test_split_negative_axis(%arg0 : tensor<16x32x64xbf16>) -> (tensor<16x16x64xbf16>, tensor<16x16x64xbf16>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = -2 : si64} : (tensor<16x32x64xbf16>, none) -> (tensor<16x16x64xbf16>, tensor<16x16x64xbf16>)
  "func.return"(%0, %1) : (tensor<16x16x64xbf16>, tensor<16x16x64xbf16>) -> ()
}

// CHECK-LABEL:  func.func @test_split_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xbf16>) -> (tensor<16x16x64xbf16>, tensor<16x16x64xbf16>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[16, 16, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<16x32x64xbf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x16x64xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[0, 16, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_4_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_3_]], [[VAR_1_]] : (tensor<16x32x64xbf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x16x64xbf16>
// CHECK:           return [[VAR_2_]], [[VAR_4_]] : tensor<16x16x64xbf16>, tensor<16x16x64xbf16>

// -----

func.func @test_non_constant_split(%arg0 : tensor<16x32x64xi16>, %arg1 : tensor<2xi64>) -> tensor<16x?x64xi16> {
  %0, %1 = "onnx.Split"(%arg0, %arg1) {axis = 1 : si64} : (tensor<16x32x64xi16>, tensor<2xi64>) -> (tensor<16x?x64xi16>, tensor<16x?x64xi16>)
  "func.return"(%0) : (tensor<16x?x64xi16>) -> ()
}

// CHECK-LABEL: func.func @test_non_constant_split
// CHECK-NOT:   tosa.slice

// -----

func.func @test_zero_split(%arg0 : tensor<16x32x64xi16>) -> tensor<16x0x64xi16> {
  %split = "onnx.Constant"() {value = dense<[32, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xi16>, tensor<2xi64>) -> (tensor<16x32x64xi16>, tensor<16x0x64xi16>)
  "func.return"(%1) : (tensor<16x0x64xi16>) -> ()
}

// CHECK-LABEL:  func.func @test_zero_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xi16>) -> tensor<16x0x64xi16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[16, 32, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {value = dense<[0, 32, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[16, 0, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_4_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_2_]], [[VAR_3_]] : (tensor<16x32x64xi16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<16x0x64xi16>
// CHECK:           return [[VAR_4_]] : tensor<16x0x64xi16>

// -----
// Legalization won't happen since tosa.slice doesn't
// allow dynamic entry in 'size' attribute
func.func @test_dynamic_shapes(%arg0 : tensor<16x32x?xf32>) -> tensor<16x2x?xf32> {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x?xf32>, tensor<2xi64>) -> (tensor<16x2x?xf32>, tensor<16x30x?xf32>)
  return %0 : tensor<16x2x?xf32>
}

// CHECK-LABEL:  func.func @test_dynamic_shapes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x?xf32>) -> tensor<16x2x?xf32> {
// CHECK-NOT:    tosa.slice

// -----
func.func @test_num_outputs(%arg0 : tensor<16x32x64xf32>) -> tensor<8x32x64xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0, %1 = "onnx.Split"(%arg0, %cst) {axis = 0 : si64, num_outputs = 2 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<8x32x64xf32>, tensor<8x32x64xf32>)
    return %0 : tensor<8x32x64xf32>
}

// CHECK-LABEL:  func.func @test_num_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x32x64xf32>) -> tensor<8x32x64xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.const_shape  {value = dense<0> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {value = dense<[8, 32, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.slice [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<16x32x64xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x32x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {value = dense<[8, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           return [[VAR_2_]] : tensor<8x32x64xf32>