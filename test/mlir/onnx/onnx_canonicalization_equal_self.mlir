// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize %s --split-input-file | FileCheck %s --check-prefixes=CHECK,STRICT
// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --canonicalize %s --split-input-file | FileCheck %s --check-prefixes=CHECK,UNSAFE

func.func @integer(%x: tensor<2x3xi32>) -> tensor<2x3xi1> {
  %r = "onnx.Equal"(%x, %x) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  return %r : tensor<2x3xi1>
}
// CHECK-LABEL: func.func @integer(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<true> : tensor<2x3xi1>
// CHECK-NOT: "onnx.Equal"
// CHECK: return [[R]]

// -----

func.func @unsigned(%x: tensor<2xui8>) -> tensor<2xi1> {
  %r = "onnx.Equal"(%x, %x) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}
// CHECK-LABEL: func.func @unsigned(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<true> : tensor<2xi1>
// CHECK-NOT: "onnx.Equal"
// CHECK: return [[R]]

// -----

func.func @float(%x: tensor<2xf32>) -> tensor<2xi1> {
  %r = "onnx.Equal"(%x, %x) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}
// CHECK-LABEL: func.func @float(
// STRICT: [[R:%[^ ]+]] = "onnx.Equal"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<true>
// CHECK: return [[R]]

// -----

func.func @dynamic(%x: tensor<?xi32>) -> tensor<?xi1> {
  %r = "onnx.Equal"(%x, %x) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
  return %r : tensor<?xi1>
}
// CHECK-LABEL: func.func @dynamic(
// CHECK: [[R:%[^ ]+]] = "onnx.Equal"
// CHECK: return [[R]]

// -----

func.func @different(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi1> {
  %r = "onnx.Equal"(%x, %y) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}
// CHECK-LABEL: func.func @different(
// CHECK: [[R:%[^ ]+]] = "onnx.Equal"
// CHECK: return [[R]]

// -----

func.func @nan() -> tensor<2xi1> {
  %x = onnx.Constant dense<0x7FC00000> : tensor<2xf32>
  %r = "onnx.Equal"(%x, %x) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}
// CHECK-LABEL: func.func @nan(
// STRICT: [[R:%[^ ]+]] = "onnx.Equal"
// UNSAFE: [[R:%[^ ]+]] = onnx.Constant dense<true>
// CHECK: return [[R]]

// -----

func.func @quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2xi1> {
  %r = "onnx.Equal"(%x, %x) : (tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2xi1>
  return %r : tensor<2xi1>
}
// CHECK-LABEL: func.func @quantized(
// CHECK: [[R:%[^ ]+]] = "onnx.Equal"
// CHECK: return [[R]]
