// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s

func.func @min_signed(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %r = "onnx.Min"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @min_signed(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_unsigned(%x: tensor<2xui64>) -> tensor<2xui64> {
  %c = onnx.Constant dense<18446744073709551615> : tensor<1xui64>
  %r = "onnx.Min"(%c, %x) : (tensor<1xui64>, tensor<2xui64>) -> tensor<2xui64>
  return %r : tensor<2xui64>
}
// CHECK-LABEL: func.func @min_unsigned(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_half(%x: tensor<2xf16>) -> tensor<2xf16> {
  %c = onnx.Constant dense<3.0> : tensor<1xf16>
  %r = "onnx.Min"(%c, %x) : (tensor<1xf16>, tensor<2xf16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// CHECK-LABEL: func.func @min_half(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_bfloat(%x: tensor<2xbf16>) -> tensor<2xbf16> {
  %c = onnx.Constant dense<3.0> : tensor<1xbf16>
  %r = "onnx.Min"(%c, %x) : (tensor<1xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
  return %r : tensor<2xbf16>
}
// CHECK-LABEL: func.func @min_bfloat(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_float(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<3.0> : tensor<1xf32>
  %r = "onnx.Min"(%c, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @min_float(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_double(%x: tensor<2xf64>) -> tensor<2xf64> {
  %c = onnx.Constant dense<3.0> : tensor<1xf64>
  %r = "onnx.Min"(%c, %x) : (tensor<1xf64>, tensor<2xf64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @min_double(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_nan(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<0x7FC00000> : tensor<1xf32>
  %r = "onnx.Min"(%c, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @min_nan(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_negative_zero(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<-0.0> : tensor<1xf32>
  %r = "onnx.Min"(%c, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @min_negative_zero(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>> {
  %c = "onnx.Constant"() {value = dense<3> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 0.5:3>>
  %r = "onnx.Min"(%c, %x) : (tensor<1x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.5:3>>
}
// CHECK-LABEL: func.func @min_quantized(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_already_normalized(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %r = "onnx.Min"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @min_already_normalized(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<3>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_both_constants(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %d = onnx.Constant dense<5> : tensor<2xi32>
  %r = "onnx.Min"(%c, %d) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @min_both_constants(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<3>
// CHECK: [[D:%[^ ]+]] = onnx.Constant dense<5>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[C]], [[D]])
// CHECK: return [[R]]

// -----

func.func @min_variadic(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %r = "onnx.Min"(%c, %x, %y) : (tensor<1xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @min_variadic(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<3>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, %arg1, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_signed(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %r = "onnx.Max"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @max_signed(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_unsigned(%x: tensor<2xui64>) -> tensor<2xui64> {
  %c = onnx.Constant dense<18446744073709551615> : tensor<1xui64>
  %r = "onnx.Max"(%c, %x) : (tensor<1xui64>, tensor<2xui64>) -> tensor<2xui64>
  return %r : tensor<2xui64>
}
// CHECK-LABEL: func.func @max_unsigned(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_half(%x: tensor<2xf16>) -> tensor<2xf16> {
  %c = onnx.Constant dense<3.0> : tensor<1xf16>
  %r = "onnx.Max"(%c, %x) : (tensor<1xf16>, tensor<2xf16>) -> tensor<2xf16>
  return %r : tensor<2xf16>
}
// CHECK-LABEL: func.func @max_half(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_bfloat(%x: tensor<2xbf16>) -> tensor<2xbf16> {
  %c = onnx.Constant dense<3.0> : tensor<1xbf16>
  %r = "onnx.Max"(%c, %x) : (tensor<1xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
  return %r : tensor<2xbf16>
}
// CHECK-LABEL: func.func @max_bfloat(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_float(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<3.0> : tensor<1xf32>
  %r = "onnx.Max"(%c, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @max_float(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_double(%x: tensor<2xf64>) -> tensor<2xf64> {
  %c = onnx.Constant dense<3.0> : tensor<1xf64>
  %r = "onnx.Max"(%c, %x) : (tensor<1xf64>, tensor<2xf64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @max_double(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_nan(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<0x7FC00000> : tensor<1xf32>
  %r = "onnx.Max"(%c, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @max_nan(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_negative_zero(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<-0.0> : tensor<1xf32>
  %r = "onnx.Max"(%c, %x) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @max_negative_zero(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>> {
  %c = "onnx.Constant"() {value = dense<3> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 0.5:3>>
  %r = "onnx.Max"(%c, %x) : (tensor<1x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.5:3>>
}
// CHECK-LABEL: func.func @max_quantized(
// CHECK: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_already_normalized(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %r = "onnx.Max"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @max_already_normalized(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<3>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_both_constants(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %d = onnx.Constant dense<5> : tensor<2xi32>
  %r = "onnx.Max"(%c, %d) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @max_both_constants(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<3>
// CHECK: [[D:%[^ ]+]] = onnx.Constant dense<5>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[C]], [[D]])
// CHECK: return [[R]]

// -----

func.func @max_variadic(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %r = "onnx.Max"(%c, %x, %y) : (tensor<1xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @max_variadic(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<3>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, %arg1, [[C]])
// CHECK: return [[R]]

// -----

func.func @min_constant_like(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = arith.constant dense<3> : tensor<1xi32>
  %r = "onnx.Min"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @min_constant_like(
// CHECK: [[C:%[^ ]+]] = arith.constant dense<3>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK: return [[R]]

// -----

func.func @max_constant_like(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = arith.constant dense<3> : tensor<1xi32>
  %r = "onnx.Max"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @max_constant_like(
// CHECK: [[C:%[^ ]+]] = arith.constant dense<3>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK: return [[R]]
