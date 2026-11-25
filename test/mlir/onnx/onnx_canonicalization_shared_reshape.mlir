// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s

func.func @shared_reshape(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Sub"(%rx, %ry) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}
// CHECK-LABEL:  func.func @shared_reshape(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[VAR_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           return [[VAR_2_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @scalar_lhs(%x: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %scalar = onnx.Constant dense<2.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Sub"(%scalar, %rx) : (tensor<f32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}

// CHECK-LABEL:  func.func @scalar_lhs(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[VAR_1_]], [[PARAM_0_]]) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @different_input_shapes(%x: tensor<2x3xf32>, %y: tensor<3x2xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<3x2xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Sub"(%rx, %ry) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}

// CHECK-LABEL:  func.func @different_input_shapes(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<3x2xf32>) -> tensor<6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<3x2xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_2_]]) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @variadic_max(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %max = "onnx.Max"(%rx, %ry, %zero) : (tensor<6xf32>, tensor<6xf32>, tensor<f32>) -> tensor<6xf32>
  return %max : tensor<6xf32>
}

// CHECK-LABEL:  func.func @variadic_max(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Max"([[VAR_2_]], [[VAR_3_]], [[VAR_1_]]) : (tensor<6xf32>, tensor<6xf32>, tensor<f32>) -> tensor<6xf32>
// CHECK:           return [[VAR_4_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @shared_reshape_min(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Min"(%rx, %ry) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}

// CHECK-LABEL:  func.func @shared_reshape_min(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Min"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[VAR_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           return [[VAR_2_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @scalar_min(%x: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %scalar = onnx.Constant dense<2.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Min"(%scalar, %rx) : (tensor<f32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}

// CHECK-LABEL:  func.func @scalar_min(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Min"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @shared_reshape_max(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Max"(%rx, %ry) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}

// CHECK-LABEL:  func.func @shared_reshape_max(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Max"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[VAR_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           return [[VAR_2_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @scalar_max(%x: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %scalar = onnx.Constant dense<2.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Max"(%scalar, %rx) : (tensor<f32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub : tensor<6xf32>
}

// CHECK-LABEL:  func.func @scalar_max(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Max"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @variadic_min(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<6xf32> {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %zero = onnx.Constant dense<0.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %max = "onnx.Min"(%rx, %ry, %zero) : (tensor<6xf32>, tensor<6xf32>, tensor<f32>) -> tensor<6xf32>
  return %max : tensor<6xf32>
}

// CHECK-LABEL:  func.func @variadic_min(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xf32>) -> tensor<6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Min"([[VAR_2_]], [[VAR_3_]], [[VAR_1_]]) : (tensor<6xf32>, tensor<6xf32>, tensor<f32>) -> tensor<6xf32>
// CHECK:           return [[VAR_4_]] : tensor<6xf32>
// CHECK:         }
// -----

func.func @reshape_with_other_user(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %ry = "onnx.Reshape"(%y, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Sub"(%rx, %ry) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub, %rx : tensor<6xf32>, tensor<6xf32>
}

// CHECK-LABEL:  func.func @reshape_with_other_user(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>, [[PARAM_1_:%.+]]: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_2_]]) : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]], [[VAR_1_]] : tensor<6xf32>, tensor<6xf32>
// CHECK:         }

// -----

func.func @singleton_rhs(%x: tensor<1x1xf32>, %y: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %r = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %result = "onnx.Sub"(%y, %r) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  return %result, %r : tensor<1x1xf32>, tensor<1xf32>
}

// CHECK-LABEL:  func.func @singleton_rhs(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x1xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[PARAM_0_]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           return [[VAR_2_]], [[VAR_1_]] : tensor<1x1xf32>, tensor<1xf32>
// CHECK:         }

// -----

func.func @singleton_lhs(%x: tensor<1x1xf32>, %y: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %r = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %result = "onnx.Sub"(%r, %y) : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result, %r : tensor<1x1xf32>, tensor<1xf32>
}

// CHECK-LABEL:  func.func @singleton_lhs(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x1xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           return [[VAR_2_]], [[VAR_1_]] : tensor<1x1xf32>, tensor<1xf32>
// CHECK:         }

// -----

func.func @singleton_min(%x: tensor<1x1xf32>, %y: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %r = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %result = "onnx.Min"(%y, %r) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  return %result, %r : tensor<1x1xf32>, tensor<1xf32>
}

// CHECK-LABEL:  func.func @singleton_min(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x1xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Min"([[PARAM_1_]], [[PARAM_0_]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           return [[VAR_2_]], [[VAR_1_]] : tensor<1x1xf32>, tensor<1xf32>
// CHECK:         }

// -----

func.func @singleton_max(%x: tensor<1x1xf32>, %y: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
  %shape = onnx.Constant dense<1> : tensor<1xi64>
  %r = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
  %result = "onnx.Max"(%y, %r) : (tensor<1x1xf32>, tensor<1xf32>) -> tensor<1x1xf32>
  return %result, %r : tensor<1x1xf32>, tensor<1xf32>
}

// CHECK-LABEL:  func.func @singleton_max(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x1xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Max"([[PARAM_1_]], [[PARAM_0_]]) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:           return [[VAR_2_]], [[VAR_1_]] : tensor<1x1xf32>, tensor<1xf32>
// CHECK:         }

// -----

func.func @singleton_rank_change(%x: tensor<1x1xf32>, %y: tensor<1x1xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>) {
  %shape = onnx.Constant dense<[1, 1, 1]> : tensor<3xi64>
  %r = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
  %result = "onnx.Sub"(%y, %r) : (tensor<1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  return %result, %r : tensor<1x1x1xf32>, tensor<1x1x1xf32>
}

// CHECK-LABEL:  func.func @singleton_rank_change(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<1x1xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>) -> (tensor<1x1x1xf32>, tensor<1x1x1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<1x1xf32>, tensor<3xi64>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[VAR_1_]]) : (tensor<1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
// CHECK:           return [[VAR_2_]], [[VAR_1_]] : tensor<1x1x1xf32>, tensor<1x1x1xf32>
// CHECK:         }

// -----

func.func @shared_reshape_scalar_lhs(%x: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %scalar = onnx.Constant dense<2.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Sub"(%scalar, %rx) : (tensor<f32>, tensor<6xf32>) -> tensor<6xf32>
  return %sub, %rx : tensor<6xf32>, tensor<6xf32>
}

// CHECK-LABEL:  func.func @shared_reshape_scalar_lhs(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_2_]]) : (tensor<f32>, tensor<6xf32>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]], [[VAR_2_]] : tensor<6xf32>, tensor<6xf32>
// CHECK:         }

// -----

func.func @shared_reshape_scalar_rhs(%x: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
  %shape = onnx.Constant dense<6> : tensor<1xi64>
  %scalar = onnx.Constant dense<2.0> : tensor<f32>
  %rx = "onnx.Reshape"(%x, %shape) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
  %sub = "onnx.Sub"(%rx, %scalar) : (tensor<6xf32>, tensor<f32>) -> tensor<6xf32>
  return %sub, %rx : tensor<6xf32>, tensor<6xf32>
}

// CHECK-LABEL:  func.func @shared_reshape_scalar_rhs(
// CHECK-SAME:   [[PARAM_0_:%.+]]: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<6xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<6xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_1_]]) : (tensor<6xf32>, tensor<f32>) -> tensor<6xf32>
// CHECK:           return [[VAR_3_]], [[VAR_2_]] : tensor<6xf32>, tensor<6xf32>
// CHECK:         }
