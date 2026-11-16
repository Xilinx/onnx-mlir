// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s --check-prefixes=CHECK,STRICT
// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s --check-prefixes=CHECK,UNSAFE

func.func @integer(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %l = onnx.Constant dense<-2> : tensor<1xi32>
  %u = onnx.Constant dense<5> : tensor<1xi32>
  %m = "onnx.Max"(%x, %l) : (tensor<2x3xi32>, tensor<1xi32>) -> tensor<2x3xi32>
  %r = "onnx.Min"(%m, %u) : (tensor<2x3xi32>, tensor<1xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @integer(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<-2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<5> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK-NOT: "onnx.Min"
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @swapped(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %l = onnx.Constant dense<-2> : tensor<1x3xi32>
  %u = onnx.Constant dense<5> : tensor<1x3xi32>
  %m = "onnx.Max"(%l, %x) : (tensor<1x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %r = "onnx.Min"(%u, %m) : (tensor<1x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @swapped(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<-2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<5> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK-NOT: "onnx.Min"
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @crossed_bounds(%x: tensor<2xi32>) -> tensor<2xi32> {
  %l = onnx.Constant dense<5> : tensor<i32>
  %u = onnx.Constant dense<-2> : tensor<i32>
  %m = "onnx.Max"(%x, %l) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Min"(%m, %u) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @crossed_bounds(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<-2> : tensor<2xi32>
// CHECK-NOT: "onnx.Min"
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @unsigned(%x: tensor<2xui64>) -> tensor<2xui64> {
  %l = onnx.Constant dense<3> : tensor<ui64>
  %u = onnx.Constant dense<18446744073709551615> : tensor<ui64>
  %m = "onnx.Max"(%x, %l) : (tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
  %r = "onnx.Min"(%m, %u) : (tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
  return %r : tensor<2xui64>
}
// CHECK-LABEL: func.func @unsigned(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<3> : tensor<ui64>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<18446744073709551615> : tensor<ui64>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK-NOT: "onnx.Min"
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @dynamic(%x: tensor<?x3xi32>) -> tensor<?x3xi32> {
  %l = onnx.Constant dense<-2> : tensor<1xi32>
  %u = onnx.Constant dense<5> : tensor<1xi32>
  %m = "onnx.Max"(%x, %l) : (tensor<?x3xi32>, tensor<1xi32>) -> tensor<?x3xi32>
  %r = "onnx.Min"(%m, %u) : (tensor<?x3xi32>, tensor<1xi32>) -> tensor<?x3xi32>
  return %r : tensor<?x3xi32>
}
// CHECK-LABEL: func.func @dynamic(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<-2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<5> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK-NOT: "onnx.Min"
// CHECK-NOT: "onnx.Max"
// CHECK: return [[R]]

// -----

func.func @float(%x: tensor<2xf32>) -> tensor<2xf32> {
  %l = onnx.Constant dense<-2.0> : tensor<f32>
  %u = onnx.Constant dense<5.0> : tensor<f32>
  %m = "onnx.Max"(%x, %l) : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  %r = "onnx.Min"(%m, %u) : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @float(
// STRICT: "onnx.Max"
// STRICT: [[R:%[^ ]+]] = "onnx.Min"
// UNSAFE: [[R:%[^ ]+]] = "onnx.Clip"(%arg0,
// CHECK: return [[R]]

// -----

func.func @double(%x: tensor<2xf64>) -> tensor<2xf64> {
  %l = onnx.Constant dense<-1.0E100> : tensor<f64>
  %u = onnx.Constant dense<1.0E100> : tensor<f64>
  %m = "onnx.Max"(%l, %x) : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
  %r = "onnx.Min"(%u, %m) : (tensor<f64>, tensor<2xf64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @double(
// STRICT: "onnx.Max"
// STRICT: [[R:%[^ ]+]] = "onnx.Min"
// UNSAFE: [[R:%[^ ]+]] = "onnx.Clip"(%arg0,
// CHECK: return [[R]]

// -----

func.func @per_channel(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %l = onnx.Constant dense<[0, 1, 2]> : tensor<3xi32>
  %u = onnx.Constant dense<5> : tensor<i32>
  %m = "onnx.Max"(%x, %l) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  %r = "onnx.Min"(%m, %u) : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @per_channel(
// CHECK: "onnx.Max"
// CHECK: [[R:%[^ ]+]] = "onnx.Min"
// CHECK: return [[R]]

// -----

func.func @expanding_broadcast(%x: tensor<2x1xi32>) -> tensor<2x3xi32> {
  %l = onnx.Constant dense<0> : tensor<1x3xi32>
  %u = onnx.Constant dense<5> : tensor<i32>
  %m = "onnx.Max"(%x, %l) : (tensor<2x1xi32>, tensor<1x3xi32>) -> tensor<2x3xi32>
  %r = "onnx.Min"(%m, %u) : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @expanding_broadcast(
// CHECK: "onnx.Max"
// CHECK: [[R:%[^ ]+]] = "onnx.Min"
// CHECK: return [[R]]

// -----

func.func @shared_max(%x: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %l = onnx.Constant dense<0> : tensor<i32>
  %u = onnx.Constant dense<5> : tensor<i32>
  %m = "onnx.Max"(%x, %l) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Min"(%m, %u) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %m, %r : tensor<2xi32>, tensor<2xi32>
}
// CHECK-LABEL: func.func @shared_max(
// CHECK: "onnx.Max"
// CHECK: "onnx.Min"
// CHECK-NOT: "onnx.Clip"

// -----

func.func @variadic(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %l = onnx.Constant dense<0> : tensor<i32>
  %u = onnx.Constant dense<5> : tensor<i32>
  %m = "onnx.Max"(%x, %l) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Min"(%m, %u, %y) : (tensor<2xi32>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @variadic(
// CHECK: "onnx.Max"
// CHECK: [[R:%[^ ]+]] = "onnx.Min"
// CHECK: return [[R]]

// -----

func.func @variable_lower(%x: tensor<2xi32>, %bound: tensor<i32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %m = "onnx.Max"(%x, %bound) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Min"(%m, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @variable_lower(
// CHECK-NOT: "onnx.Clip"
// CHECK: [[M:%[^ ]+]] = "onnx.Max"
// CHECK-NOT: "onnx.Clip"
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[M]],
// CHECK-NOT: "onnx.Clip"
// CHECK: return [[R]]

// -----

func.func @variable_upper(%x: tensor<2xi32>, %bound: tensor<i32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %m = "onnx.Max"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Min"(%m, %bound) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @variable_upper(
// CHECK-NOT: "onnx.Clip"
// CHECK: [[M:%[^ ]+]] = "onnx.Max"
// CHECK-NOT: "onnx.Clip"
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[M]],
// CHECK-NOT: "onnx.Clip"
// CHECK: return [[R]]

// -----

func.func @quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>> {
  %f = onnx.Constant dense<0.0> : tensor<2xf32>
  %c = "quant.qcast"(%f) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  %m = "onnx.Max"(%x, %c) : (tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  %r = "onnx.Min"(%m, %c) : (tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.5:3>>
}
// CHECK-LABEL: func.func @quantized(
// CHECK: "onnx.Max"
// CHECK: [[R:%[^ ]+]] = "onnx.Min"
// CHECK: return [[R]]
