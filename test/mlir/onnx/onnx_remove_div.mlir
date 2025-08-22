// RUN: onnx-mlir-opt --dq-binary-q-opt-onnx-to-onnx %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_removebinary_pattern1
// CHECK-NOT: onnx.Div
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern1(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%39 = onnx.Constant dense<0.152590215> : tensor<f32>

%961 = "onnx.DequantizeLinear"(%36, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1180 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1182 = "onnx.Div"(%1180, %961) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>

%1184 = "onnx.QuantizeLinear"(%1182, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1186 = "onnx.DequantizeLinear"(%1184, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %1186 : tensor<1x1x1x128xf32>
}

//-----

// CHECK-LABEL: func.func @test_removebinary_pattern2
// CHECK-NOT: onnx.Div
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern2(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%36 = onnx.Constant dense<0> : tensor<ui16>
%37 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%38 = onnx.Constant dense<65535> : tensor<ui16>
%39 = onnx.Constant dense<0.152590215> : tensor<f32>

%961 = "onnx.DequantizeLinear"(%36, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>

%1180 = "onnx.DequantizeLinear"(%arg0, %37, %36) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

%1182 = "onnx.Div"(%1180, %961) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>

%1184 = "onnx.QuantizeLinear"(%1182, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64,
output_dtype = 0 : si64,
saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>

%1186 = "onnx.DequantizeLinear"(%1184, %39, %38) {
axis = 1 : si64,
block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>

return %1186 : tensor<1x1x1x128xf32>
}
