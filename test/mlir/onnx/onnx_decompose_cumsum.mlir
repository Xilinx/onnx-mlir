// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

func.func @cumsum_static_shape_v1(%arg0: tensor<8x1x384xf32>, %arg1: tensor<i64>) -> (tensor<8x1x384xf32>) {
    %Y = "onnx.CumSum"(%arg0, %arg1) {onnx_node_name = "/CumSum"} : (tensor<8x1x384xf32>, tensor<i64>) -> tensor<8x1x384xf32>
    return %Y : tensor<8x1x384xf32>
// CHECK-LABEL: func.func @cumsum_static_shape_v1
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<8x1x384xf32>, [[PARAM_1_:%.+]]: tensor<i64>) -> tensor<8x1x384xf32> {
// CHECK-NEXT: [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT: [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NEXT: [[VAR_2_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-NEXT: [[VAR_3_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-NEXT: [[VAR_4_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-NEXT: [[VAR_5_:%.+]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK-NEXT: [[VAR_6_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-NEXT: [[VAR_7_:%.+]] = onnx.Constant dense<7> : tensor<1xi64>
// CHECK-NEXT: [[VAR_8_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>

// CHECK: [[VAR_S0_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S1_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S2_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_2_]], [[VAR_3_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_3_]], [[VAR_4_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S4_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S5_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_5_]], [[VAR_6_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S6_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_6_]], [[VAR_7_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_S7_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_7_]], [[VAR_8_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x1x384xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x1x384xf32>

// CHECK: [[VAR_A0_:%.+]] = "onnx.Add"([[VAR_S0_]], [[VAR_S1_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_A1_:%.+]] = "onnx.Add"([[VAR_A0_]], [[VAR_S2_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_A2_:%.+]] = "onnx.Add"([[VAR_A1_]], [[VAR_S3_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_A3_:%.+]] = "onnx.Add"([[VAR_A2_]], [[VAR_S4_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_A4_:%.+]] = "onnx.Add"([[VAR_A3_]], [[VAR_S5_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_A5_:%.+]] = "onnx.Add"([[VAR_A4_]], [[VAR_S6_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>
// CHECK: [[VAR_A6_:%.+]] = "onnx.Add"([[VAR_A5_]], [[VAR_S7_]])
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<1x1x384xf32>

// CHECK: [[Y_:%.+]] = "onnx.Concat"([[VAR_S0_]], [[VAR_A0_]], [[VAR_A1_]], [[VAR_A2_]], [[VAR_A3_]], [[VAR_A4_]], [[VAR_A5_]], [[VAR_A6_]])
// CHECK-SAME: axis = 0 : si64
// CHECK-SAME: : (tensor<1x1x384xf32>, tensor<1x1x384xf32>, tensor<1x1x384xf32>, tensor<1x1x384xf32>, tensor<1x1x384xf32>, tensor<1x1x384xf32>, tensor<1x1x384xf32>, tensor<1x1x384xf32>) -> tensor<8x1x384xf32>
// CHECK: return [[Y_]] : tensor<8x1x384xf32>
}

// -----

// Edge case - When the axis 0 dimension is 1, CumSum is redundant
func.func @cumsum_static_shape_v2(%arg0: tensor<1x1x384xf32>, %arg1: tensor<i64>) -> (tensor<1x1x384xf32>) {
    %Y = "onnx.CumSum"(%arg0, %arg1) {onnx_node_name = "/CumSum"} : (tensor<1x1x384xf32>, tensor<i64>) -> tensor<1x1x384xf32>
    return %Y : tensor<1x1x384xf32>
// CHECK-LABEL: func.func @cumsum_static_shape_v2
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<1x1x384xf32>, [[PARAM_1_:%.+]]: tensor<i64>) -> tensor<1x1x384xf32> {
//CHECK-NEXT: return [[PARAM_0_]] : tensor<1x1x384xf32>
}

// -----

func.func @cumsum_dynamic_shape_v1(%arg0: tensor<8x?x?xf32>, %arg1: tensor<i64>) -> (tensor<8x?x?xf32>) {
    %Y = "onnx.CumSum"(%arg0, %arg1) {onnx_node_name = "/CumSum"} : (tensor<8x?x?xf32>, tensor<i64>) -> tensor<8x?x?xf32>
    return %Y : tensor<8x?x?xf32>
// CHECK-LABEL: func.func @cumsum_dynamic_shape_v1
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<8x?x?xf32>, [[PARAM_1_:%.+]]: tensor<i64>) -> tensor<8x?x?xf32> {
// CHECK-NEXT: [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-NEXT: [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NEXT: [[VAR_2_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-NEXT: [[VAR_3_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-NEXT: [[VAR_4_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-NEXT: [[VAR_5_:%.+]] = onnx.Constant dense<5> : tensor<1xi64>
// CHECK-NEXT: [[VAR_6_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-NEXT: [[VAR_7_:%.+]] = onnx.Constant dense<7> : tensor<1xi64>
// CHECK-NEXT: [[VAR_8_:%.+]] = onnx.Constant dense<8> : tensor<1xi64>

// CHECK: [[VAR_S0_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S1_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S2_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_2_]], [[VAR_3_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S3_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_3_]], [[VAR_4_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S4_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S5_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_5_]], [[VAR_6_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S6_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_6_]], [[VAR_7_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_S7_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_7_]], [[VAR_8_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME: : (tensor<8x?x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x?x?xf32>

// CHECK: [[VAR_A0_:%.+]] = "onnx.Add"([[VAR_S0_]], [[VAR_S1_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_A1_:%.+]] = "onnx.Add"([[VAR_A0_]], [[VAR_S2_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_A2_:%.+]] = "onnx.Add"([[VAR_A1_]], [[VAR_S3_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_A3_:%.+]] = "onnx.Add"([[VAR_A2_]], [[VAR_S4_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_A4_:%.+]] = "onnx.Add"([[VAR_A3_]], [[VAR_S5_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_A5_:%.+]] = "onnx.Add"([[VAR_A4_]], [[VAR_S6_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
// CHECK: [[VAR_A6_:%.+]] = "onnx.Add"([[VAR_A5_]], [[VAR_S7_]])
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>

// CHECK: [[Y_:%.+]] = "onnx.Concat"([[VAR_S0_]], [[VAR_A0_]], [[VAR_A1_]], [[VAR_A2_]], [[VAR_A3_]], [[VAR_A4_]], [[VAR_A5_]], [[VAR_A6_]])
// CHECK-SAME: axis = 0 : si64
// CHECK-SAME: : (tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<8x?x?xf32>
// CHECK: return [[Y_]] : tensor<8x?x?xf32>
}

// -----

// Batch dimension is required for this decomposition. If it is not available then the pass should simply fail and
// do nothing to the input mlir.
func.func @cumsum_dynamic_shape_v2(%arg0: tensor<*xf32>, %arg1: tensor<i64>) -> (tensor<*xf32>) {
    %Y = "onnx.CumSum"(%arg0, %arg1) {onnx_node_name = "/CumSum"} : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
    return %Y : tensor<*xf32>
// CHECK-LABEL: func.func @cumsum_dynamic_shape_v2
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<i64>) -> tensor<*xf32> {
// CHECK-NEXT: [[Y_:%.+]] = "onnx.CumSum"([[PARAM_0_]], [[PARAM_1_]])
// CHECK-SAME: : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
// CHECK-NEXT: return [[Y_]] : tensor<*xf32>
}

// -----

// Batch dimension is required for this decomposition. If it is not available then the pass should simply fail and
// do nothing to the input mlir.
func.func @cumsum_dynamic_shape_v3(%arg0: tensor<?x1x384xf32>, %arg1: tensor<i64>) -> (tensor<?x1x384xf32>) {
    %Y = "onnx.CumSum"(%arg0, %arg1) {onnx_node_name = "/CumSum"} : (tensor<?x1x384xf32>, tensor<i64>) -> tensor<?x1x384xf32>
    return %Y : tensor<?x1x384xf32>
// CHECK-LABEL: func.func @cumsum_dynamic_shape_v3
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<?x1x384xf32>, [[PARAM_1_:%.+]]: tensor<i64>) -> tensor<?x1x384xf32> {
// CHECK-NEXT: [[Y_:%.+]] = "onnx.CumSum"([[PARAM_0_]], [[PARAM_1_]])
// CHECK-SAME: : (tensor<?x1x384xf32>, tensor<i64>) -> tensor<?x1x384xf32>
// CHECK-NEXT: return [[Y_]] : tensor<?x1x384xf32>
}
