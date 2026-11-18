// (c) Copyright 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not='"onnx.{{(Div|LeakyRelu|PRelu|Selu|Softplus|ThresholdedRelu|Clip|BatchNormalizationInferenceMode|QuantizeLinear|DequantizeLinear|Gemm|Softmax|ReduceMean|ReduceMeanV13|Gather|Shrink|Sqrt|Elu|HardSigmoid)}}"' --implicit-check-not='"onnx.{{(Add|Sub|Mul|Pow|Min|Max|Equal|Greater|GreaterOrEqual)}}"'
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Add,Sub,Mul,Pow,Min,Max,Equal,Greater,GreaterOrEqual" -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,EXCLUDED --implicit-check-not=tosa.add --implicit-check-not=tosa.sub --implicit-check-not=tosa.mul --implicit-check-not=tosa.pow --implicit-check-not=tosa.minimum --implicit-check-not=tosa.maximum --implicit-check-not=tosa.equal --implicit-check-not=tosa.greater --implicit-check-not=tosa.greater_equal --implicit-check-not='"onnx.{{(Div|LeakyRelu|PRelu|Selu|Softplus|ThresholdedRelu|Clip|BatchNormalizationInferenceMode|QuantizeLinear|DequantizeLinear|Gemm|Softmax|ReduceMean|ReduceMeanV13|Gather|Shrink|Sqrt|Elu|HardSigmoid)}}"'
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa="excluded-ops=Add,Sub,Mul,Pow,Min,Max,Equal,Greater,GreaterOrEqual" --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s --check-prefixes=CHECK,LOWERED --implicit-check-not='"onnx.Add"' --implicit-check-not='"onnx.Sub"' --implicit-check-not='"onnx.Mul"' --implicit-check-not='"onnx.Pow"' --implicit-check-not='"onnx.Min"' --implicit-check-not='"onnx.Max"' --implicit-check-not='"onnx.Equal"' --implicit-check-not='"onnx.Greater"' --implicit-check-not='"onnx.GreaterOrEqual"' --implicit-check-not='"onnx.{{(Div|LeakyRelu|PRelu|Selu|Softplus|ThresholdedRelu|Clip|BatchNormalizationInferenceMode|QuantizeLinear|DequantizeLinear|Gemm|Softmax|ReduceMean|ReduceMeanV13|Gather|Shrink|Sqrt|Elu|HardSigmoid)}}"' --implicit-check-not='"onnx.{{(Add|Sub|Mul|Pow|Min|Max|Equal|Greater|GreaterOrEqual)}}"'

func.func @intermediate_div_decomposed_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
}

// CHECK-LABEL: func.func @intermediate_div_decomposed_broadcast(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_div_dynamic_float(%arg0: tensor<?x?x?xf32>, %arg1: tensor<13x?x?xf32>) -> tensor<13x?x?xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<13x?x?xf32>) -> tensor<13x?x?xf32>
  "func.return"(%0) : (tensor<13x?x?xf32>) -> ()
}

// CHECK-LABEL: func.func @intermediate_div_dynamic_float(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.707330704 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: func.func @intermediate_leaky_relu(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// EXCLUDED-DAG: "onnx.GreaterOrEqual"
// LOWERED-DAG: tosa.greater_equal
// CHECK: return

// -----

func.func @intermediate_prelu(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: func.func @intermediate_prelu(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// EXCLUDED-DAG: "onnx.GreaterOrEqual"
// LOWERED-DAG: tosa.greater_equal
// CHECK: return

// -----

func.func @intermediate_selu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha = 1.5 : f32, gamma = 2.0 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: func.func @intermediate_selu(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// EXCLUDED-DAG: "onnx.Sub"
// LOWERED-DAG: tosa.sub
// EXCLUDED-DAG: "onnx.Greater"
// LOWERED-DAG: tosa.greater
// CHECK: return

// -----

func.func @intermediate_softplus_dynamic(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  func.return %0 : tensor<?x3x?xf32>
}

// CHECK-LABEL: func.func @intermediate_softplus_dynamic(
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// CHECK: return

// -----

func.func @intermediate_thresholdedrelu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.ThresholdedRelu"(%arg0) {alpha = 0.5 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
}

// CHECK-LABEL: func.func @intermediate_thresholdedrelu(
// EXCLUDED-DAG: "onnx.Greater"
// LOWERED-DAG: tosa.greater
// CHECK: return

// -----

func.func @intermediate_clip(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<3xi32> {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xi32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: func.func @intermediate_clip(
// EXCLUDED-DAG: "onnx.Max"
// LOWERED-DAG: tosa.maximum
// EXCLUDED-DAG: "onnx.Min"
// LOWERED-DAG: tosa.minimum
// CHECK: return

// -----

func.func @intermediate_batchnorm_f32(%arg0: tensor<100x3x10x10xf32>) -> tensor<100x3x10x10xf32> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf32>} : () -> tensor<3xf32>
    %4 = "onnx.BatchNormalizationInferenceMode"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x10x10xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<100x3x10x10xf32>
    return %4 : tensor<100x3x10x10xf32>
}

// CHECK-LABEL: func.func @intermediate_batchnorm_f32(
// EXCLUDED-DAG: "onnx.Sub"
// LOWERED-DAG: tosa.sub
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_quantizeLinear_per_axis(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xf32>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}

// CHECK-LABEL: func.func @intermediate_quantizeLinear_per_axis(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// CHECK: return

// -----

func.func @intermediate_per_axis(%arg0: tensor<8x2xi8>) -> tensor<8x2xf32> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xi8>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xf32>
  return %2 : tensor<8x2xf32>
}

// CHECK-LABEL: func.func @intermediate_per_axis(
// EXCLUDED-DAG: "onnx.Sub"
// LOWERED-DAG: tosa.sub
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32>  {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @intermediate_alpha(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// CHECK: return

// -----

func.func @intermediate_softmax_v13(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
}

// CHECK-LABEL: func.func @intermediate_softmax_v13(
// EXCLUDED-DAG: "onnx.Sub"
// LOWERED-DAG: tosa.sub
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
}

// CHECK-LABEL: func.func @intermediate_reduce_mean(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_reducemeanV13(%arg0: tensor<1x32x112x112xf32>) -> tensor<1x32x1x1xf32> {
  %0 = "onnx.ReduceMeanV13"(%arg0) {axes = [2, 3], keepdims = 1 : si64} : (tensor<1x32x112x112xf32>) -> tensor<1x32x1x1xf32>
  return %0 : tensor<1x32x1x1xf32>
}

// CHECK-LABEL: func.func @intermediate_reducemeanV13(
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return

// -----

func.func @intermediate_gather_dynamic_indices_i32(%arg0 : tensor<3x3xf32>, %indices: tensor<1x2xi32>) -> tensor<3x1x2xf32> {
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi32>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()

}

// CHECK-LABEL: func.func @intermediate_gather_dynamic_indices_i32(
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// EXCLUDED-DAG: "onnx.GreaterOrEqual"
// LOWERED-DAG: tosa.greater_equal
// CHECK: return

// -----

func.func @intermediate_shrink_float(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %2 = "onnx.Shrink"(%arg0) {lambd = -7.500000e-01 : f32, bias = 5.000000e-01 : f32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @intermediate_shrink_float(
// EXCLUDED-DAG: "onnx.Greater"
// LOWERED-DAG: tosa.greater
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// EXCLUDED-DAG: "onnx.Sub"
// LOWERED-DAG: tosa.sub
// CHECK: return

// -----

func.func @sqrt(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Sqrt"(%input) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @sqrt(
// EXCLUDED-DAG: "onnx.Pow"
// LOWERED-DAG: tosa.pow
// CHECK: return

// -----

func.func @elu(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.Elu"(%input) {alpha = 0.5 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @elu(
// EXCLUDED-DAG: "onnx.Sub"
// LOWERED-DAG: tosa.sub
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// EXCLUDED-DAG: "onnx.GreaterOrEqual"
// LOWERED-DAG: tosa.greater_equal
// CHECK: return

// -----

func.func @hard_sigmoid(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %r = "onnx.HardSigmoid"(%input) {alpha = 0.2 : f32, beta = 0.5 : f32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @hard_sigmoid(
// EXCLUDED-DAG: "onnx.Add"
// LOWERED-DAG: tosa.add
// EXCLUDED-DAG: "onnx.Mul"
// LOWERED-DAG: tosa.mul
// CHECK: return
