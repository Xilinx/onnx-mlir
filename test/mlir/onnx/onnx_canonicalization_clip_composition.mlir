// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" --constprop-onnx --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s

func.func @disjoint_above(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<0> : tensor<i32>
  %b1 = onnx.Constant dense<1> : tensor<i32>
  %b2 = onnx.Constant dense<2> : tensor<i32>
  %b3 = onnx.Constant dense<3> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @disjoint_above(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<2> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @disjoint_below(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<2> : tensor<i32>
  %b1 = onnx.Constant dense<3> : tensor<i32>
  %b2 = onnx.Constant dense<0> : tensor<i32>
  %b3 = onnx.Constant dense<1> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @disjoint_below(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @overlap(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<0> : tensor<i32>
  %b1 = onnx.Constant dense<5> : tensor<i32>
  %b2 = onnx.Constant dense<2> : tensor<i32>
  %b3 = onnx.Constant dense<7> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @overlap(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<5> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK: return [[R]]

// -----

func.func @outer_inside(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<0> : tensor<i32>
  %b1 = onnx.Constant dense<9> : tensor<i32>
  %b2 = onnx.Constant dense<2> : tensor<i32>
  %b3 = onnx.Constant dense<7> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @outer_inside(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<7> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK: return [[R]]

// -----

func.func @inner_inside(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<2> : tensor<i32>
  %b1 = onnx.Constant dense<7> : tensor<i32>
  %b2 = onnx.Constant dense<0> : tensor<i32>
  %b3 = onnx.Constant dense<9> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @inner_inside(
// CHECK-DAG: [[L:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK-DAG: [[U:%[^ ]+]] = onnx.Constant dense<7> : tensor<i32>
// CHECK: [[R:%[^ ]+]] = "onnx.Clip"(%arg0, [[L]], [[U]])
// CHECK: return [[R]]

// -----

func.func @inner_lower_only(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<2> : tensor<i32>
  %b2 = onnx.Constant dense<0> : tensor<i32>
  %b3 = onnx.Constant dense<1> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %none) : (tensor<2xi32>, tensor<i32>, none) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @inner_lower_only(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @inner_upper_only(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b1 = onnx.Constant dense<1> : tensor<i32>
  %b2 = onnx.Constant dense<2> : tensor<i32>
  %b3 = onnx.Constant dense<3> : tensor<i32>
  %a = "onnx.Clip"(%x, %none, %b1) : (tensor<2xi32>, none, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @inner_upper_only(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<2> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @outer_lower_only(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<0> : tensor<i32>
  %b1 = onnx.Constant dense<1> : tensor<i32>
  %b2 = onnx.Constant dense<2> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %none) : (tensor<2xi32>, tensor<i32>, none) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @outer_lower_only(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<2> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @outer_upper_only(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<2> : tensor<i32>
  %b1 = onnx.Constant dense<3> : tensor<i32>
  %b3 = onnx.Constant dense<1> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %none, %b3) : (tensor<2xi32>, none, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @outer_upper_only(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<1> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @outer_crossed(%x: tensor<2xi32>) -> tensor<2xi32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %b0 = onnx.Constant dense<0> : tensor<i32>
  %b1 = onnx.Constant dense<9> : tensor<i32>
  %b2 = onnx.Constant dense<7> : tensor<i32>
  %b3 = onnx.Constant dense<2> : tensor<i32>
  %a = "onnx.Clip"(%x, %b0, %b1) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Clip"(%a, %b2, %b3) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @outer_crossed(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<2> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @minmax_after_clip(%x: tensor<2xi32>) -> tensor<2xi32> {
  %zero = onnx.Constant dense<0> : tensor<i32>
  %one = onnx.Constant dense<1> : tensor<i32>
  %two = onnx.Constant dense<2> : tensor<i32>
  %three = onnx.Constant dense<3> : tensor<i32>
  %y = "onnx.Clip"(%x, %zero, %one) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %m = "onnx.Max"(%y, %two) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  %r = "onnx.Min"(%m, %three) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @minmax_after_clip(
// CHECK: [[R:%[^ ]+]] = onnx.Constant dense<2> : tensor<2xi32>
// CHECK: return [[R]]

// -----

func.func @where_after_clip(%x: tensor<2xi32>) -> tensor<2xi32> {
  %zero = onnx.Constant dense<0> : tensor<i32>
  %one = onnx.Constant dense<1> : tensor<i32>
  %two = onnx.Constant dense<2> : tensor<i32>
  %three = onnx.Constant dense<3> : tensor<i32>
  %y = "onnx.Clip"(%x, %zero, %one) : (tensor<2xi32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %p = "onnx.GreaterOrEqual"(%y, %two) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  %r = "onnx.Where"(%p, %y, %two) : (tensor<2xi1>, tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}
// CHECK-LABEL: func.func @where_after_clip(
// CHECK-DAG: [[TWO:%[^ ]+]] = onnx.Constant dense<2> : tensor<i32>
// CHECK: [[Y:%[^ ]+]] = "onnx.Clip"(%arg0,
// CHECK: [[R:%[^ ]+]] = "onnx.Max"([[Y]], [[TWO]])
// CHECK: return [[R]]
