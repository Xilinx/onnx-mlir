// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize --constprop-onnx --canonicalize %s --split-input-file | FileCheck %s

func.func @signed_lower() -> tensor<2xi32> {
  %x = onnx.Constant dense<[5, 2147483647]> : tensor<2xi32>
  %c = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @signed_lower(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<[5, 2147483647]> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @signed_upper() -> tensor<2xi32> {
  %x = onnx.Constant dense<[-5, -2147483648]> : tensor<2xi32>
  %c = onnx.Constant dense<-3> : tensor<i32>
  %p = "onnx.LessOrEqual"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @signed_upper(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<[-5, -2147483648]> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @signed64_lower() -> tensor<2xi64> {
  %x = onnx.Constant dense<[5, 9223372036854775807]> : tensor<2xi64>
  %c = onnx.Constant dense<3> : tensor<i64>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi64>, tensor<i64>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
  return %r : tensor<2xi64>
}
// CHECK-LABEL: func.func @signed64_lower(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<[5, 9223372036854775807]> : tensor<2xi64>
// CHECK: return [[R]]

// -----

func.func @signed64_upper() -> tensor<2xi64> {
  %x = onnx.Constant dense<[-5, -9223372036854775808]> : tensor<2xi64>
  %c = onnx.Constant dense<-3> : tensor<i64>
  %p = "onnx.LessOrEqual"(%x, %c) : (tensor<2xi64>, tensor<i64>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
  return %r : tensor<2xi64>
}
// CHECK-LABEL: func.func @signed64_upper(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<[-5, -9223372036854775808]> : tensor<2xi64>
// CHECK: return [[R]]

// -----

func.func @unsigned_lower() -> tensor<2xui32> {
  %x = onnx.Constant dense<[5, 4294967295]> : tensor<2xui32>
  %c = onnx.Constant dense<3> : tensor<ui32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
  return %r : tensor<2xui32>
}
// CHECK-LABEL: func.func @unsigned_lower(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<[5, 4294967295]> : tensor<2xui32>
// CHECK: return [[R]]

// -----

func.func @unsigned_upper() -> tensor<2xui32> {
  %x = onnx.Constant dense<[0, 5]> : tensor<2xui32>
  %c = onnx.Constant dense<7> : tensor<ui32>
  %p = "onnx.LessOrEqual"(%x, %c) : (tensor<2xui32>, tensor<ui32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xui32>, tensor<ui32>) -> tensor<2xui32>
  return %r : tensor<2xui32>
}
// CHECK-LABEL: func.func @unsigned_upper(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<[0, 5]> : tensor<2xui32>
// CHECK: return [[R]]
