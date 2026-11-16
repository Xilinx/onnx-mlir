// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize %s --split-input-file | FileCheck %s --check-prefixes=CHECK,STRICT
// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --canonicalize %s --split-input-file | FileCheck %s --check-prefixes=CHECK,UNSAFE

func.func @integer_broadcast(%x: tensor<2x1xi32>) -> tensor<2x3xi32> {
  %z = onnx.Constant dense<0> : tensor<1x3xi32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1x3xi32>, tensor<2x1xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @integer_broadcast(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<0> : tensor<2x3xi32>
// CHECK-NOT: "onnx.Mul"
// CHECK: return [[R]]

// -----

func.func @unsigned(%x: tensor<2xui8>) -> tensor<2xui8> {
  %z = onnx.Constant dense<0> : tensor<2xui8>
  %r = "onnx.Mul"(%z, %x) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
  return %r : tensor<2xui8>
}
// CHECK-LABEL: func.func @unsigned(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<0> : tensor<2xui8>
// CHECK-NOT: "onnx.Mul"
// CHECK: return [[R]]

// -----

func.func @float_broadcast(%x: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %z = onnx.Constant dense<0.0> : tensor<1x3xf32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @float_broadcast(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK: return [[R]]

// -----

func.func @double(%x: tensor<2xf64>) -> tensor<2xf64> {
  %z = onnx.Constant dense<-0.0> : tensor<2xf64>
  %r = "onnx.Mul"(%z, %x) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @double(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf64>
// CHECK: return [[R]]

// -----

func.func @dynamic(%x: tensor<?xi32>) -> tensor<?xi32> {
  %z = onnx.Constant dense<0> : tensor<1xi32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}
// CHECK-LABEL: func.func @dynamic(
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"
// CHECK: return [[R]]

// -----

func.func @nonzero(%x: tensor<2xi32>) -> tensor<2xi32> {
  %z = onnx.Constant dense<2> : tensor<1xi32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @nonzero(
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"
// CHECK: return [[R]]

// -----

func.func @integer_rhs_zero(%x: tensor<2x1xi32>) -> tensor<2x3xi32> {
  %z = onnx.Constant dense<0> : tensor<1x3xi32>
  %r = "onnx.Mul"(%x, %z) : (tensor<2x1xi32>, tensor<1x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @integer_rhs_zero(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<0> : tensor<2x3xi32>
// CHECK-NOT: "onnx.Mul"
// CHECK: return [[R]]

// -----

func.func @float_rhs_zero(%x: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %z = onnx.Constant dense<0.0> : tensor<1x3xf32>
  %r = "onnx.Mul"(%x, %z) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @float_rhs_zero(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2x3xf32>
// CHECK: return [[R]]

// -----

func.func @nan() -> tensor<2xf32> {
  %x = onnx.Constant dense<0x7FC00000> : tensor<2xf32>
  %z = onnx.Constant dense<0.0> : tensor<1xf32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @nan(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: return [[R]]

// -----

func.func @infinity() -> tensor<2xf32> {
  %x = onnx.Constant dense<0x7F800000> : tensor<2xf32>
  %z = onnx.Constant dense<0.0> : tensor<1xf32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @infinity(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: return [[R]]

// -----

func.func @negative() -> tensor<2xf32> {
  %x = onnx.Constant dense<-1.0> : tensor<2xf32>
  %z = onnx.Constant dense<0.0> : tensor<1xf32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @negative(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: return [[R]]

// -----

func.func @zero() -> tensor<2xf32> {
  %x = onnx.Constant dense<0.0> : tensor<2xf32>
  %z = onnx.Constant dense<0.0> : tensor<1xf32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @zero(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: return [[R]]


// -----

func.func @scalar(%x: tensor<i64>) -> tensor<i64> {
  %z = onnx.Constant dense<0> : tensor<i64>
  %r = "onnx.Mul"(%z, %x) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  return %r : tensor<i64>
}
// CHECK-LABEL: func.func @scalar(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<0> : tensor<i64>
// CHECK: return [[R]]

// -----

func.func @half(%x: tensor<2xf16>) -> tensor<2xf16> {
  %z = onnx.Constant dense<0.0> : tensor<1xf16>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xf16>, tensor<2xf16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// CHECK-LABEL: func.func @half(
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf16>
// CHECK: return [[R]]

// -----

func.func @unranked(%x: tensor<*xi32>) -> tensor<*xi32> {
  %z = onnx.Constant dense<0> : tensor<1xi32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xi32>, tensor<*xi32>) -> tensor<*xi32>
  return %r : tensor<*xi32>
}
// CHECK-LABEL: func.func @unranked(
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"
// CHECK: return [[R]]

// -----

func.func @empty(%x: tensor<0xi32>) -> tensor<0xi32> {
  %z = onnx.Constant dense<0> : tensor<1xi32>
  %r = "onnx.Mul"(%z, %x) : (tensor<1xi32>, tensor<0xi32>) -> tensor<0xi32>
  return %r : tensor<0xi32>
}
// CHECK-LABEL: func.func @empty(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<0> : tensor<0xi32>
// CHECK: return [[R]]


// -----

func.func @qdq(%x: tensor<2xi8>) -> tensor<2xi8> {
  %s = onnx.Constant dense<0.5> : tensor<f32>
  %zp = onnx.Constant dense<3> : tensor<i8>
  %dq = "onnx.DequantizeLinear"(%x, %s, %zp) : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
  %z = onnx.Constant dense<0.0> : tensor<1xf32>
  %r = "onnx.Mul"(%z, %dq) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  %q = "onnx.QuantizeLinear"(%r, %s, %zp) : (tensor<2xf32>, tensor<f32>, tensor<i8>) -> tensor<2xi8>
  return %q : tensor<2xi8>
}
// CHECK-LABEL: func.func @qdq(
// STRICT: "onnx.DequantizeLinear"
// STRICT: [[R:%[^ ]+]] = "onnx.Mul"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<0.000000e+00> : tensor<2xf32>
// CHECK: [[Q:%[^ ]+]] = "onnx.QuantizeLinear"([[R]],
// CHECK: return [[Q]]

// -----

func.func @quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>> {
  %f = onnx.Constant dense<0.0> : tensor<2xf32>
  %z = "quant.qcast"(%f) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  %r = "onnx.Mul"(%z, %x) : (tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.5:3>>
}
// CHECK-LABEL: func.func @quantized(
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"
// CHECK: return [[R]]
