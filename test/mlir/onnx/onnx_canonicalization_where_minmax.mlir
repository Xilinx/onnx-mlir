// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s

func.func @greaterorequal_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @greaterorequal_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @greaterorequal_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @greaterorequal_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @greaterorequal_reversed_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @greaterorequal_reversed_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @greaterorequal_reversed_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @greaterorequal_reversed_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @lessorequal_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.LessOrEqual"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @lessorequal_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @lessorequal_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.LessOrEqual"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @lessorequal_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @lessorequal_reversed_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.LessOrEqual"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @lessorequal_reversed_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @lessorequal_reversed_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.LessOrEqual"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @lessorequal_reversed_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @unsigned(%x: tensor<2xui64>) -> tensor<2xui64> {
  %c = onnx.Constant dense<18446744073709551615> : tensor<ui64>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xui64>, tensor<ui64>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
  return %r : tensor<2xui64>
}
// CHECK-LABEL: func.func @unsigned(
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0,
// CHECK: return [[R]]

// -----

func.func @float(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @float(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @dynamic(%x: tensor<?xi32>) -> tensor<?xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<?xi1>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}
// CHECK-LABEL: func.func @dynamic(
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0,
// CHECK: return [[R]]

// -----

func.func @different_bound(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %v = onnx.Constant dense<4> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @different_bound(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @unrelated_input(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %y, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @unrelated_input(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @per_channel(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<[3, 4]> : tensor<2xi32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @per_channel(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @expanding_broadcast(%x: tensor<2x1xi32>) -> tensor<2x3xi32> {
  %c = onnx.Constant dense<3> : tensor<1x3xi32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2x1xi32>, tensor<1x3xi32>) -> tensor<2x3xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2x3xi1>, tensor<2x1xi32>, tensor<1x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @expanding_broadcast(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @nan() -> tensor<2xf32> {
  %x = onnx.Constant dense<0x7FC00000> : tensor<2xf32>
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %c, %x) : (tensor<2xi1>, tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @nan(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @infinity() -> tensor<2xf32> {
  %x = onnx.Constant dense<0x7F800000> : tensor<2xf32>
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %c, %x) : (tensor<2xi1>, tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @infinity(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @negative_zero() -> tensor<2xf32> {
  %x = onnx.Constant dense<-0.0> : tensor<2xf32>
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %c, %x) : (tensor<2xi1>, tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @negative_zero(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>> {
  %f = onnx.Constant dense<0.0> : tensor<2xf32>
  %c = "quant.qcast"(%f) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  %p = "onnx.GreaterOrEqual"(%x, %c) : (tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.5:3>>
}
// CHECK-LABEL: func.func @quantized(
// CHECK: "onnx.GreaterOrEqual"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @paired_bounds(%x: tensor<2xi32>) -> tensor<2xi32> {
  %lower = onnx.Constant dense<2> : tensor<i32>
  %upper = onnx.Constant dense<7> : tensor<i32>
  %lp = "onnx.GreaterOrEqual"(%x, %lower) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %lo = "onnx.Where"(%lp, %x, %lower) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %up = "onnx.LessOrEqual"(%lo, %upper) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%up, %lo, %upper) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @paired_bounds(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<7> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @constant_input_reversed_greater_equal() -> tensor<2xi32> {
  %c = onnx.Constant dense<2> : tensor<2xi32>
  %x = onnx.Constant dense<[1, 4]> : tensor<2xi32>
  %p = "onnx.GreaterOrEqual"(%c, %x) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @constant_input_reversed_greater_equal(
// CHECK-DAG: [[X:%[^ ]+]] = onnx.Constant dense<[1, 4]> : tensor<2xi32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[X]], [[C]])
// CHECK: return [[R]]

// -----

func.func @constant_input_reversed_less_equal() -> tensor<2xi32> {
  %c = onnx.Constant dense<2> : tensor<2xi32>
  %x = onnx.Constant dense<[1, 4]> : tensor<2xi32>
  %p = "onnx.LessOrEqual"(%c, %x) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @constant_input_reversed_less_equal(
// CHECK-DAG: [[X:%[^ ]+]] = onnx.Constant dense<[1, 4]> : tensor<2xi32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[X]], [[C]])
// CHECK: return [[R]]

// -----

func.func @strict_greaterorequal_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_greaterorequal_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_greaterorequal_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_greaterorequal_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_greaterorequal_reversed_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Greater"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_greaterorequal_reversed_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_greaterorequal_reversed_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Greater"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_greaterorequal_reversed_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_lessorequal_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Less"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_lessorequal_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_lessorequal_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Less"(%x, %c) : (tensor<2xi32>, tensor<1xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_lessorequal_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_lessorequal_reversed_input_true(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Less"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_lessorequal_reversed_input_true(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_lessorequal_reversed_input_false(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<1xi32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Less"(%c, %x) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %v, %x) : (tensor<2xi1>, tensor<i32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_lessorequal_reversed_input_false(
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<3> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"(%arg0, [[C]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_unsigned(%x: tensor<2xui64>) -> tensor<2xui64> {
  %c = onnx.Constant dense<18446744073709551615> : tensor<ui64>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xui64>, tensor<ui64>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xui64>, tensor<ui64>) -> tensor<2xui64>
  return %r : tensor<2xui64>
}
// CHECK-LABEL: func.func @strict_unsigned(
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0,
// CHECK: return [[R]]

// -----

func.func @strict_float(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @strict_float(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_dynamic(%x: tensor<?xi32>) -> tensor<?xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Greater"(%x, %c) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<?xi1>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %r : tensor<?xi32>
}
// CHECK-LABEL: func.func @strict_dynamic(
// CHECK: [[R:%[^ ]+]] = "onnx.Max"(%arg0,
// CHECK: return [[R]]

// -----

func.func @strict_different_bound(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %v = onnx.Constant dense<4> : tensor<i32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_different_bound(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_unrelated_input(%x: tensor<2xi32>, %y: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<3> : tensor<i32>
  %v = onnx.Constant dense<3> : tensor<i32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %y, %v) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_unrelated_input(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_per_channel(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = onnx.Constant dense<[3, 4]> : tensor<2xi32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_per_channel(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_expanding_broadcast(%x: tensor<2x1xi32>) -> tensor<2x3xi32> {
  %c = onnx.Constant dense<3> : tensor<1x3xi32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2x1xi32>, tensor<1x3xi32>) -> tensor<2x3xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2x3xi1>, tensor<2x1xi32>, tensor<1x3xi32>) -> tensor<2x3xi32>
  return %r : tensor<2x3xi32>
}
// CHECK-LABEL: func.func @strict_expanding_broadcast(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_nan() -> tensor<2xf32> {
  %x = onnx.Constant dense<0x7FC00000> : tensor<2xf32>
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %c, %x) : (tensor<2xi1>, tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @strict_nan(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_infinity() -> tensor<2xf32> {
  %x = onnx.Constant dense<0x7F800000> : tensor<2xf32>
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %c, %x) : (tensor<2xi1>, tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @strict_infinity(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_negative_zero() -> tensor<2xf32> {
  %x = onnx.Constant dense<-0.0> : tensor<2xf32>
  %c = onnx.Constant dense<0.0> : tensor<f32>
  %p = "onnx.Greater"(%x, %c) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %c, %x) : (tensor<2xi1>, tensor<f32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @strict_negative_zero(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_quantized(%x: tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>> {
  %f = onnx.Constant dense<0.0> : tensor<2xf32>
  %c = "quant.qcast"(%f) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  %p = "onnx.Greater"(%x, %c) : (tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>, tensor<2x!quant.uniform<i8:f32, 0.5:3>>) -> tensor<2x!quant.uniform<i8:f32, 0.5:3>>
  return %r : tensor<2x!quant.uniform<i8:f32, 0.5:3>>
}
// CHECK-LABEL: func.func @strict_quantized(
// CHECK: "onnx.Greater"
// CHECK: [[R:%[^ ]+]] = "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_paired_bounds(%x: tensor<2xi32>) -> tensor<2xi32> {
  %lower = onnx.Constant dense<2> : tensor<i32>
  %upper = onnx.Constant dense<7> : tensor<i32>
  %lp = "onnx.Greater"(%x, %lower) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %lo = "onnx.Where"(%lp, %x, %lower) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %up = "onnx.Less"(%lo, %upper) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%up, %lo, %upper) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_paired_bounds(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<7> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK-NOT: "onnx.Where"
// CHECK: return [[R]]

// -----

func.func @strict_constant_input_reversed_greater_equal() -> tensor<2xi32> {
  %c = onnx.Constant dense<2> : tensor<2xi32>
  %x = onnx.Constant dense<[1, 4]> : tensor<2xi32>
  %p = "onnx.Greater"(%c, %x) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_constant_input_reversed_greater_equal(
// CHECK-DAG: [[X:%[^ ]+]] = onnx.Constant dense<[1, 4]> : tensor<2xi32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Min"([[X]], [[C]])
// CHECK: return [[R]]

// -----

func.func @strict_constant_input_reversed_less_equal() -> tensor<2xi32> {
  %c = onnx.Constant dense<2> : tensor<2xi32>
  %x = onnx.Constant dense<[1, 4]> : tensor<2xi32>
  %p = "onnx.Less"(%c, %x) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %x, %c) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @strict_constant_input_reversed_less_equal(
// CHECK-DAG: [[X:%[^ ]+]] = onnx.Constant dense<[1, 4]> : tensor<2xi32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[X]], [[C]])
// CHECK: return [[R]]
