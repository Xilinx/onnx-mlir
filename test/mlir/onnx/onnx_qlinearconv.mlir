// RUN: onnx-mlir-opt --conv-to-qlinearconv-onnx-to-onnx %s -split-input-file | FileCheck %s

 
  func.func @test_bias_int32(%arg0: tensor<1x2x4x4xi8>) -> tensor<1x4x4x4xi8> {

    %scaleX = onnx.Constant dense<2.500000e-01> : tensor<f32>
    %zpX    = onnx.Constant dense<0> : tensor<i8>

    %Wf = onnx.Constant dense<[
      [[ [1.0] ], [ [2.0] ]],
      [[ [3.0] ], [ [4.0] ]],
      [[ [5.0] ], [ [6.0] ]],
      [[ [7.0] ], [ [8.0] ]]
    ]> : tensor<4x2x1x1xf32>
    %scaleW = onnx.Constant dense<7.812500e-03> : tensor<f32>
    %zpW    = onnx.Constant dense<0> : tensor<i8>
    %qW = "onnx.QuantizeLinear"(%Wf, %scaleW, %zpW)
          : (tensor<4x2x1x1xf32>, tensor<f32>, tensor<i8>) -> tensor<4x2x1x1xi8>
    %dqW = "onnx.DequantizeLinear"(%qW, %scaleW, %zpW)
           : (tensor<4x2x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<4x2x1x1xf32>

    %bias_i32 = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xi32>
    %scaleB   = onnx.Constant dense<6.250000e-02> : tensor<f32>
    %zpB      = onnx.Constant dense<0> : tensor<i32>
    %dqB = "onnx.DequantizeLinear"(%bias_i32, %scaleB, %zpB)
           : (tensor<4xi32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>

    %dqX = "onnx.DequantizeLinear"(%arg0, %scaleX, %zpX)
           : (tensor<1x2x4x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x2x4x4xf32>

    %conv = "onnx.Conv"(%dqX, %dqW, %dqB)
            {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0]}
            : (tensor<1x2x4x4xf32>, tensor<4x2x1x1xf32>, tensor<4xf32>) -> tensor<1x4x4x4xf32>

    %scaleY = onnx.Constant dense<2.500000e-01> : tensor<f32>
    %zpY    = onnx.Constant dense<0> : tensor<i8>
    %qOut = "onnx.QuantizeLinear"(%conv, %scaleY, %zpY)
            : (tensor<1x4x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4x4x4xi8>

    return %qOut : tensor<1x4x4x4xi8>
  }


// CHECK: "onnx.QLinearConv"
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.QuantizeLinear"

func.func @test_qlinearconv(%arg0: tensor<1x3x16x16xi8>) -> tensor<1x8x14x14xi8> {
    %scaleX = "onnx.Constant"() {value = dense<0.00784314> : tensor<f32>} : () -> tensor<f32>
    %zpX    = "onnx.Constant"() {value = dense<0> : tensor<i8>}           : () -> tensor<i8>
    %w      = "onnx.Constant"() {value = dense<0> : tensor<8x3x3x3xi8>}   : () -> tensor<8x3x3x3xi8>
    %scaleW = "onnx.Constant"() {value = dense<0.05> : tensor<f32>}       : () -> tensor<f32>
    %zpW    = "onnx.Constant"() {value = dense<0> : tensor<i8>}           : () -> tensor<i8>
    %biasF  = "onnx.Constant"() {value = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : tensor<8xf32>} : () -> tensor<8xf32>
    %biasQ  = "onnx.QuantizeLinear"(%biasF, %scaleW, %zpW)
                {axis = 1 : si64} : (tensor<8xf32>, tensor<f32>, tensor<i8>) -> tensor<8xi8>
    %biasDQ = "onnx.DequantizeLinear"(%biasQ, %scaleW, %zpW)
                {axis = 1 : si64} : (tensor<8xi8>, tensor<f32>, tensor<i8>) -> tensor<8xf32>
    %dqX    = "onnx.DequantizeLinear"(%arg0, %scaleX, %zpX)
                {axis = 1 : si64} : (tensor<1x3x16x16xi8>, tensor<f32>, tensor<i8>) -> tensor<1x3x16x16xf32>
    %dqW    = "onnx.DequantizeLinear"(%w, %scaleW, %zpW)
                {axis = 1 : si64} : (tensor<8x3x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<8x3x3x3xf32>
    %conv   = "onnx.Conv"(%dqX, %dqW, %biasDQ) {kernel_shape = [3,3]} :
                (tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>) -> tensor<1x8x14x14xf32>
    %scaleY = "onnx.Constant"() {value = dense<0.09> : tensor<f32>} : () -> tensor<f32>
    %zpY    = "onnx.Constant"() {value = dense<0> : tensor<i8>}     : () -> tensor<i8>
    %qOut   = "onnx.QuantizeLinear"(%conv, %scaleY, %zpY)
                {axis = 1 : si64} : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<i8>) -> tensor<1x8x14x14xi8>
    return %qOut : tensor<1x8x14x14xi8>
  }
 
// CHECK: "onnx.QLinearConv"
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.QuantizeLinear"

func.func @test_onnx_conv_without_quantize(%arg0: tensor<1x24x127x127xi8>) -> tensor<1x144x127x127xf32> {
  %0 = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = onnx.Constant dense_resource<__elided__> : tensor<144x24x1x1xf32>
  %3 = onnx.Constant dense<7.812500e-03> : tensor<f32>
  %4 = onnx.Constant dense<0> : tensor<i8>
  %5 = onnx.Constant dense<7.812500e-03> : tensor<f32>
  %6 = onnx.Constant dense<0> : tensor<i8>
  %7 = onnx.Constant dense_resource<__elided__> : tensor<144xf32>
  %8 = onnx.Constant dense<6.250000e-02> : tensor<f32>
  %9 = onnx.Constant dense<0> : tensor<i8>
  %10 = onnx.Constant dense<6.250000e-02> : tensor<f32>
  %11 = onnx.Constant dense<0> : tensor<i8>
  %12 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %13 = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %14 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "DequantizeLinear_72"} : (tensor<1x24x127x127xi8>, tensor<f32>, tensor<i8>) -> tensor<1x24x127x127xf32>
  %15 = "onnx.QuantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "QuantizeLinear_75", output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<144x24x1x1xf32>, tensor<f32>, tensor<i8>) -> tensor<144x24x1x1xi8>
  %16 = "onnx.DequantizeLinear"(%15, %5, %6) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "DequantizeLinear_78"} : (tensor<144x24x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<144x24x1x1xf32>
  %17 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "QuantizeLinear_81", output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<144xf32>, tensor<f32>, tensor<i8>) -> tensor<144xi8>
  %18 = "onnx.DequantizeLinear"(%17, %10, %11) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "DequantizeLinear_84"} : (tensor<144xi8>, tensor<f32>, tensor<i8>) -> tensor<144xf32>
  %19 = "onnx.Conv"(%14, %16, %18) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "Conv_85", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x24x127x127xf32>, tensor<144x24x1x1xf32>, tensor<144xf32>) -> tensor<1x144x127x127xf32>
  %20 = "onnx.Clip"(%19, %12, %13) {onnx_node_name = "Clip_88"} : (tensor<1x144x127x127xf32>, tensor<f32>, tensor<f32>) -> tensor<1x144x127x127xf32>
  return %20 : tensor<1x144x127x127xf32>
}

 
// CHECK-LABEL:  func.func @test_onnx_conv_without_quantize
// CHECK-NOT: onnx.QLinearConv