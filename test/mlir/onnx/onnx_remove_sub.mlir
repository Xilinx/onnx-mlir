// RUN: onnx-mlir-opt --dq-binary-q-opt-onnx-to-onnx %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_removebinary_pattern1
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern1(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%14 = onnx.Constant dense<39664> : tensor<ui16>
%15 = onnx.Constant dense<2.57987776E-5> : tensor<f32>

%960 = "onnx.DequantizeLinear"(%38, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1174 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1176 = "onnx.Sub"(%960, %1174) {onnx_node_name = "/bert/Sub"} : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%1178 = "onnx.QuantizeLinear"(%1176, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1180 = "onnx.DequantizeLinear"(%1178, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %1180 : tensor<1x1x1x128xf32>
}

// -----

// CHECK-LABEL: func.func @test_removebinary_pattern2
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern2(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%14 = onnx.Constant dense<39664> : tensor<ui16>
%15 = onnx.Constant dense<2.57987776E-5> : tensor<f32>

%960 = "onnx.DequantizeLinear"(%38, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1174 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1176 = "onnx.Sub"(%960, %1174) {onnx_node_name = "/bert/Sub"} : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>

%1178 = "onnx.QuantizeLinear"(%1176, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1180 = "onnx.DequantizeLinear"(%1178, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %1180 : tensor<1x1x1x128xf32>
}
