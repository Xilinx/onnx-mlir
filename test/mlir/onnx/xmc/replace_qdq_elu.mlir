// RUN: onnx-mlir-opt --replace-qdq-eltwise %s | FileCheck %s

// CHECK-LABEL: func.func @elu_u8_alpha
func.func @elu_u8_alpha(%input: tensor<4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<4x!quant.uniform<u8:f32, 0.1:128>> {
  %output = "onnx.Elu"(%input) {alpha = 2.0 : f32} : (tensor<4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<4x!quant.uniform<u8:f32, 0.1:128>>
  return %output : tensor<4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-NOT: "onnx.Elu"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: {alpha = 2.000000e+00 : f32, enable_lut_sigmoid = false, nonlinear = "NONE", type = "ELU"}
  // CHECK-NOT: "onnx.Elu"
  // CHECK: return
}

// CHECK-LABEL: func.func @elu_u8_default_alpha
func.func @elu_u8_default_alpha(%input: tensor<4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<4x!quant.uniform<u8:f32, 0.1:128>> {
  %output = "onnx.Elu"(%input) : (tensor<4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<4x!quant.uniform<u8:f32, 0.1:128>>
  return %output : tensor<4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-NOT: "onnx.Elu"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: {alpha = 1.000000e+00 : f32, enable_lut_sigmoid = false, nonlinear = "NONE", type = "ELU"}
  // CHECK-NOT: "onnx.Elu"
  // CHECK: return
}

// CHECK-LABEL: func.func @elu_u8_zero_alpha
func.func @elu_u8_zero_alpha(%input: tensor<4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<4x!quant.uniform<u8:f32, 0.1:128>> {
  %output = "onnx.Elu"(%input) {alpha = 0.0 : f32} : (tensor<4x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<4x!quant.uniform<u8:f32, 0.1:128>>
  return %output : tensor<4x!quant.uniform<u8:f32, 0.1:128>>

  // CHECK-NOT: "onnx.Elu"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: {alpha = 0.000000e+00 : f32, enable_lut_sigmoid = false, nonlinear = "NONE", type = "ELU"}
  // CHECK-NOT: "onnx.Elu"
  // CHECK: return
}

// CHECK-LABEL: func.func @elu_i16_alpha
func.func @elu_i16_alpha(%input: tensor<4x!quant.uniform<i16:f32, 0.0001:0>>) -> tensor<4x!quant.uniform<i16:f32, 0.0001:0>> {
  %output = "onnx.Elu"(%input) {alpha = 0.5 : f32} : (tensor<4x!quant.uniform<i16:f32, 0.0001:0>>) -> tensor<4x!quant.uniform<i16:f32, 0.0001:0>>
  return %output : tensor<4x!quant.uniform<i16:f32, 0.0001:0>>

  // CHECK-NOT: "onnx.Elu"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: {alpha = 5.000000e-01 : f32, enable_lut_sigmoid = false, nonlinear = "NONE", type = "ELU"}
  // CHECK-NOT: "onnx.Elu"
  // CHECK: return
}

// CHECK-LABEL: func.func @elu_u16_default_alpha
func.func @elu_u16_default_alpha(%input: tensor<4x!quant.uniform<u16:f32, 0.0001:32768>>) -> tensor<4x!quant.uniform<u16:f32, 0.0001:32768>> {
  %output = "onnx.Elu"(%input) : (tensor<4x!quant.uniform<u16:f32, 0.0001:32768>>) -> tensor<4x!quant.uniform<u16:f32, 0.0001:32768>>
  return %output : tensor<4x!quant.uniform<u16:f32, 0.0001:32768>>

  // CHECK-NOT: "onnx.Elu"
  // CHECK: "onnx.XCOMPILERFusedEltwise"
  // CHECK-SAME: {alpha = 1.000000e+00 : f32, enable_lut_sigmoid = false, nonlinear = "NONE", type = "ELU"}
  // CHECK-NOT: "onnx.Elu"
  // CHECK: return
}

// CHECK-LABEL: func.func @elu_float_unfused
func.func @elu_float_unfused(%input: tensor<4xf32>) -> tensor<4xf32> {
  %output = "onnx.Elu"(%input) {alpha = 2.0 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %output : tensor<4xf32>

  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
  // CHECK: "onnx.Elu"(%arg0) {alpha = 2.000000e+00 : f32}
  // CHECK-NOT: "onnx.XCOMPILERFusedEltwise"
  // CHECK: return
}
