// RUN: onnx-mlir-opt --dq-binary-q-opt %s | FileCheck %s --check-prefix=FOLD
// RUN: onnx-mlir-opt --dq-binary-q-opt --convert-qdq-to-requantize --quant-types --replace-erf-to-gelu %s | FileCheck %s --check-prefix=GELU
// RUN: %python %S/Inputs/erf_gelu_qdq.py onnx-mlir-opt %s
// RUN: onnx-mlir-opt --xmc-passes %s | FileCheck %s --check-prefix=XMC

// FOLD-LABEL: func.func @encoder_gelu
// FOLD: %[[HALF:.*]] = onnx.Constant dense<5.000000e-01>
// FOLD: %[[TAIL:.*]] = "onnx.Mul"({{.*}}, %[[HALF]])
// FOLD: "onnx.QuantizeLinear"(%[[TAIL]],
// GELU-LABEL: func.func @encoder_gelu
// GELU-NOT: "onnx.Erf"
// GELU: "onnx.Gelu"
// GELU-SAME: -> tensor<1x4x!quant.uniform<u8:f32, 0.019999999552965164:128>>
// GELU-NOT: "onnx.Mul"
// XMC-LABEL: func.func @encoder_gelu
// XMC-NOT: "onnx.Erf"
// XMC-NOT: "onnx.Div"
// XMC: "onnx.XCOMPILERFusedEltwise"
// XMC-SAME: type = "GELU"
// XMC-SAME: -> tensor<1x4x!quant.uniform<u8:f32, 0.019999999552965164:128>>
// XMC-NOT: "onnx.Mul"
func.func @encoder_gelu(%input: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %sqrt2 = onnx.Constant dense<1.41421356> : tensor<f32>
  %one = onnx.Constant dense<1.0> : tensor<f32>
  %half = onnx.Constant dense<0.5> : tensor<f32>
  %input_scale = onnx.Constant dense<0.03> : tensor<f32>
  %div_scale = onnx.Constant dense<0.021> : tensor<f32>
  %erf_scale = onnx.Constant dense<0.008> : tensor<f32>
  %output_scale = onnx.Constant dense<0.02> : tensor<f32>
  %zp = onnx.Constant dense<128> : tensor<ui8>
  %x = "onnx.DequantizeLinear"(%input, %input_scale, %zp) : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %div = "onnx.Div"(%x, %sqrt2) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  %div_q = "onnx.QuantizeLinear"(%div, %div_scale, %zp) : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  %div_dq = "onnx.DequantizeLinear"(%div_q, %div_scale, %zp) : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %erf = "onnx.Erf"(%div_dq) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %erf_q = "onnx.QuantizeLinear"(%erf, %erf_scale, %zp) : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  %erf_dq = "onnx.DequantizeLinear"(%erf_q, %erf_scale, %zp) : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
  %add = "onnx.Add"(%erf_dq, %one) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  %mul = "onnx.Mul"(%x, %add) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %tail = "onnx.Mul"(%mul, %half) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  %output = "onnx.QuantizeLinear"(%tail, %output_scale, %zp) : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %output : tensor<1x4xui8>
}
