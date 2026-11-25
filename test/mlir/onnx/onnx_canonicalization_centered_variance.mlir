// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --canonicalize-with-rn --canonicalize="test-convergence=true" --cse %s --split-input-file | FileCheck %s --check-prefixes=CHECK,UNSAFE
// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize="test-convergence=true" --cse %s --split-input-file | FileCheck %s --check-prefix=STRICT

func.func @direct(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp : tensor<1x1x256xf32>
}

// CHECK-LABEL:  func.func @direct(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
// UNSAFE:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// UNSAFE:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// UNSAFE:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x512x256xf32>, tensor<1x1x256xf32>) -> tensor<1x512x256xf32>
// UNSAFE:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// UNSAFE:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// UNSAFE:           return [[VAR_4_]] : tensor<1x1x256xf32>
// UNSAFE:         }
// STRICT-LABEL: func.func @direct(
// STRICT-NOT: func.func
// STRICT:           {{%.+}} = "onnx.Max"({{%.+}}, {{%.+}}) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// STRICT:           return {{%.+}} : tensor<1x1x256xf32>

// -----

func.func @singleton_reshape(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %shape = onnx.Constant dense<[1, 256]> : tensor<2xi64>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %second_r = "onnx.Reshape"(%second, %shape) {allowzero = 0 : si64} : (tensor<1x1x256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
  %mean_r = "onnx.Reshape"(%mean, %shape) {allowzero = 0 : si64} : (tensor<1x1x256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
  %mean_square = "onnx.Mul"(%mean_r, %mean_r) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %variance = "onnx.Sub"(%second_r, %mean_square) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x256xf32>) -> tensor<1x256xf32>
  return %clamp : tensor<1x256xf32>
}

// CHECK-LABEL:  func.func @singleton_reshape(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x256xf32> {
// UNSAFE-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// UNSAFE-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 256]> : tensor<2xi64>
// UNSAFE:           [[VAR_2_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// UNSAFE:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<1x512x256xf32>, tensor<1x1x256xf32>) -> tensor<1x512x256xf32>
// UNSAFE:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_3_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// UNSAFE:           [[VAR_5_:%.+]] = "onnx.ReduceMean"([[VAR_4_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// UNSAFE:           [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1x1x256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// UNSAFE:           return [[VAR_6_]] : tensor<1x256xf32>
// UNSAFE:         }

// -----

// V13 reductions upgrade to ReduceMean before the variance pattern matches.
func.func @v13_negative_axis(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMeanV13"(%square) {axes = [-2], keepdims = 1 : si64} : (tensor<1x512x256xf32>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [1], keepdims = 1 : si64} : (tensor<1x512x256xf32>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp : tensor<1x1x256xf32>
}

// CHECK-LABEL:  func.func @v13_negative_axis(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
// UNSAFE:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// UNSAFE:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// UNSAFE:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<1x512x256xf32>, tensor<1x1x256xf32>) -> tensor<1x512x256xf32>
// UNSAFE:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// UNSAFE:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// UNSAFE:           return [[VAR_4_]] : tensor<1x1x256xf32>
// UNSAFE:         }

// -----

func.func @nonzero_clamp(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<1.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp : tensor<1x1x256xf32>
}

// CHECK-LABEL:  func.func @nonzero_clamp(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x1x256xf32>
// CHECK:         }

// -----

func.func @different_input(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%y, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp : tensor<1x1x256xf32>
}

// CHECK-LABEL:  func.func @different_input(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_1_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x1x256xf32>
// CHECK:         }

// -----

func.func @not_a_square(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %y) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp : tensor<1x1x256xf32>
}

// CHECK-LABEL:  func.func @not_a_square(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x1x256xf32>
// CHECK:         }

// -----

func.func @non_singleton_reshape(%x: tensor<2x3x4xf32>, %y: tensor<2x3x4xf32>) -> tensor<4x2xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %shape = onnx.Constant dense<[4, 2]> : tensor<2xi64>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %second_r = "onnx.Reshape"(%second, %shape) {allowzero = 0 : si64} : (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<4x2xf32>
  %mean_r = "onnx.Reshape"(%mean, %shape) {allowzero = 0 : si64} : (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<4x2xf32>
  %mean_square = "onnx.Mul"(%mean_r, %mean_r) : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %variance = "onnx.Sub"(%second_r, %mean_square) : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %clamp : tensor<4x2xf32>
}

// CHECK-LABEL:  func.func @non_singleton_reshape(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x4xf32>) -> tensor<4x2xf32> {
// UNSAFE-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// UNSAFE-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[4, 2]> : tensor<2xi64>
// UNSAFE:           [[VAR_2_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
// UNSAFE:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<2x3x4xf32>, tensor<2x1x4xf32>) -> tensor<2x3x4xf32>
// UNSAFE:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_3_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// UNSAFE:           [[VAR_5_:%.+]] = "onnx.ReduceMean"([[VAR_4_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
// UNSAFE:           [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<4x2xf32>
// UNSAFE:           return [[VAR_6_]] : tensor<4x2xf32>
// UNSAFE:         }

// -----

func.func @different_axes(%x: tensor<2x4x4xf32>, %y: tensor<2x4x4xf32>) -> tensor<2x4xf32> {
  %other_axes = onnx.Constant dense<2> : tensor<1xi64>
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %shape = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
  %mean = "onnx.ReduceMean"(%x, %other_axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4x4xf32>, tensor<1xi64>) -> tensor<2x4x1xf32>
  %second_r = "onnx.Reshape"(%second, %shape) {allowzero = 0 : si64} : (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  %mean_r = "onnx.Reshape"(%mean, %shape) {allowzero = 0 : si64} : (tensor<2x4x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  %mean_square = "onnx.Mul"(%mean_r, %mean_r) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %variance = "onnx.Sub"(%second_r, %mean_square) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %clamp : tensor<2x4xf32>
}

// CHECK-LABEL:  func.func @different_axes(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x4x4xf32>, [[PARAM_1_:%.+]]: tensor<2x4x4xf32>) -> tensor<2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[2, 4]> : tensor<2xi64>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReduceMean"([[VAR_4_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4x4xf32>, tensor<1xi64>) -> tensor<2x1x4xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x4x4xf32>, tensor<1xi64>) -> tensor<2x4x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<2x1x4xf32>, tensor<2xi64>) -> tensor<2x4xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Mul"([[VAR_6_]], [[VAR_6_]]) : (tensor<2x4x1xf32>, tensor<2x4x1xf32>) -> tensor<2x4x1xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_8_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<2x4x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Sub"([[VAR_7_]], [[VAR_9_]]) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Max"([[VAR_10_]], [[VAR_2_]]) : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
// CHECK:           return [[VAR_11_]] : tensor<2x4xf32>
// CHECK:         }

// -----

func.func @shared_second_moment(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> (tensor<1x1x256xf32>, tensor<1x1x256xf32>) {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp, %second : tensor<1x1x256xf32>, tensor<1x1x256xf32>
}


// CHECK-LABEL:  func.func @shared_second_moment(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> (tensor<1x1x256xf32>, tensor<1x1x256xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// CHECK:           return [[VAR_7_]], [[VAR_3_]] : tensor<1x1x256xf32>, tensor<1x1x256xf32>
// CHECK:         }

// -----

func.func @shared_variance(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> (tensor<1x1x256xf32>, tensor<1x1x256xf32>) {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp, %variance : tensor<1x1x256xf32>, tensor<1x1x256xf32>
}


// CHECK-LABEL:  func.func @shared_variance(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> (tensor<1x1x256xf32>, tensor<1x1x256xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// CHECK:           return [[VAR_7_]], [[VAR_6_]] : tensor<1x1x256xf32>, tensor<1x1x256xf32>
// CHECK:         }

// -----

func.func @quantized_variance(%x: tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>, %y: tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 0.25>> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>, tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>) -> tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>, tensor<1xi64>) -> tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>, tensor<1xi64>) -> tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>, tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>, tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>
  return %clamp : tensor<1x1x256x!quant.uniform<i8:f32, 0.25>>
}


// CHECK-LABEL:  func.func @quantized_variance(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>, [[PARAM_1_:%.+]]: tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>) -> tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<1xi64>) -> tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<1xi64>) -> tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>) -> tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<f32>) -> tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           return [[VAR_7_]] : tensor<1x1x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:         }

// -----

func.func @quantized_second_moment(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256x!quant.uniform<i8:f32, 0.25>>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  return %clamp : tensor<1x1x256xf32>
}


// CHECK-LABEL:  func.func @quantized_second_moment(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256x!quant.uniform<i8:f32, 2.500000e-01>>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<f32>) -> tensor<1x1x256xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x1x256xf32>
// CHECK:         }

// -----

func.func @clamp_changes_rank(%x: tensor<1x512x256xf32>, %y: tensor<1x512x256xf32>) -> tensor<1x1x1x256xf32> {
  %axes = onnx.Constant dense<1> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<1x1x1x1xf32>
  %square = "onnx.Mul"(%x, %x) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<1x1x1x1xf32>, tensor<1x1x256xf32>) -> tensor<1x1x1x256xf32>
  return %clamp : tensor<1x1x1x256xf32>
}


// CHECK-LABEL:  func.func @clamp_changes_rank(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x512x256xf32>, [[PARAM_1_:%.+]]: tensor<1x512x256xf32>) -> tensor<1x1x1x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<1x512x256xf32>, tensor<1x512x256xf32>) -> tensor<1x512x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.ReduceMean"([[VAR_2_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x512x256xf32>, tensor<1xi64>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_4_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_3_]], [[VAR_5_]]) : (tensor<1x1x256xf32>, tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Max"([[VAR_6_]], [[VAR_1_]]) : (tensor<1x1x256xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x256xf32>
// CHECK:           return [[VAR_7_]] : tensor<1x1x1x256xf32>
// CHECK:         }

// -----

func.func @absent_axes(%x: tensor<2x3x4xf32>, %y: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  return %clamp : tensor<1x1x1xf32>
}
// CHECK-LABEL:  func.func @absent_axes(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// UNSAFE:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// UNSAFE:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// UNSAFE:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x3x4xf32>, tensor<1x1x1xf32>) -> tensor<2x3x4xf32>
// UNSAFE:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// UNSAFE:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// UNSAFE:           return [[VAR_4_]] : tensor<1x1x1xf32>
// UNSAFE:         }

// -----

func.func @empty_axes(%x: tensor<2x3x4xf32>, %y: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
  %axes = onnx.Constant dense<> : tensor<0xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<0xi64>) -> tensor<1x1x1xf32>
  %mean = "onnx.ReduceMean"(%x, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<0xi64>) -> tensor<1x1x1xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  return %clamp : tensor<1x1x1xf32>
}
// CHECK-LABEL:  func.func @empty_axes(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// UNSAFE:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
// UNSAFE:           [[VAR_1_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// UNSAFE:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x3x4xf32>, tensor<1x1x1xf32>) -> tensor<2x3x4xf32>
// UNSAFE:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_2_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// UNSAFE:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// UNSAFE:           return [[VAR_4_]] : tensor<1x1x1xf32>
// UNSAFE:         }

// -----

func.func @different_axis_order(%x: tensor<2x3x4xf32>, %y: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
  %axes = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %other_axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %square = "onnx.Mul"(%x, %x) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %second = "onnx.ReduceMean"(%square, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
  %mean = "onnx.ReduceMean"(%x, %other_axes) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
  %mean_square = "onnx.Mul"(%mean, %mean) : (tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
  %variance = "onnx.Sub"(%second, %mean_square) : (tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
  %clamp = "onnx.Max"(%zero, %variance) : (tensor<f32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
  return %clamp : tensor<2x1x1xf32>
}

// CHECK-LABEL:  func.func @different_axis_order(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x3x4xf32>) -> tensor<2x1x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReduceMean"([[PARAM_0_]], [[VAR_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x1x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_5_]], [[VAR_5_]]) : (tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Sub"([[VAR_4_]], [[VAR_6_]]) : (tensor<2x1x1xf32>, tensor<2x1x1xf32>) -> tensor<2x1x1xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Max"([[VAR_7_]], [[VAR_2_]]) : (tensor<2x1x1xf32>, tensor<f32>) -> tensor<2x1x1xf32>
// CHECK:           return [[VAR_8_]] : tensor<2x1x1xf32>
// CHECK:         }
