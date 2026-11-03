// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --unsafe-math-optimizations=true --canonicalize="test-convergence=true" %s --split-input-file | FileCheck %s

func.func @exp_log_f16(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %0 = "onnx.Log"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %1 = "onnx.Exp"(%0) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return %1 : tensor<2x3xf16>
}

// CHECK-LABEL: func.func @exp_log_f16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf16>
// CHECK-NOT: "onnx.Log"
// CHECK-NOT: "onnx.Exp"
// CHECK: return [[X]] : tensor<2x3xf16>

// -----

func.func @exp_log_bf16(%x: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  %0 = "onnx.Log"(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  %1 = "onnx.Exp"(%0) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %1 : tensor<2x3xbf16>
}

// CHECK-LABEL: func.func @exp_log_bf16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xbf16>
// CHECK-NOT: "onnx.Log"
// CHECK-NOT: "onnx.Exp"
// CHECK: return [[X]] : tensor<2x3xbf16>

// -----

func.func @exp_log_f32(%x: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Log"(%x) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %1 = "onnx.Exp"(%0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  return %1 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @exp_log_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<?x3xf32>
// CHECK-NOT: "onnx.Log"
// CHECK-NOT: "onnx.Exp"
// CHECK: return [[X]] : tensor<?x3xf32>

// -----

func.func @exp_log_f64(%x: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "onnx.Log"(%x) : (tensor<*xf64>) -> tensor<*xf64>
  %1 = "onnx.Exp"(%0) : (tensor<*xf64>) -> tensor<*xf64>
  return %1 : tensor<*xf64>
}

// CHECK-LABEL: func.func @exp_log_f64(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<*xf64>
// CHECK-NOT: "onnx.Log"
// CHECK-NOT: "onnx.Exp"
// CHECK: return [[X]] : tensor<*xf64>

// -----

func.func @log_exp_f16(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %0 = "onnx.Exp"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %1 = "onnx.Log"(%0) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return %1 : tensor<2x3xf16>
}

// CHECK-LABEL: func.func @log_exp_f16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf16>
// CHECK-NOT: "onnx.Exp"
// CHECK-NOT: "onnx.Log"
// CHECK: return [[X]] : tensor<2x3xf16>

// -----

func.func @log_exp_bf16(%x: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  %0 = "onnx.Exp"(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  %1 = "onnx.Log"(%0) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %1 : tensor<2x3xbf16>
}

// CHECK-LABEL: func.func @log_exp_bf16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xbf16>
// CHECK-NOT: "onnx.Exp"
// CHECK-NOT: "onnx.Log"
// CHECK: return [[X]] : tensor<2x3xbf16>

// -----

func.func @log_exp_f32(%x: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Exp"(%x) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %1 = "onnx.Log"(%0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  return %1 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @log_exp_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<?x3xf32>
// CHECK-NOT: "onnx.Exp"
// CHECK-NOT: "onnx.Log"
// CHECK: return [[X]] : tensor<?x3xf32>

// -----

func.func @log_exp_f64(%x: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "onnx.Exp"(%x) : (tensor<*xf64>) -> tensor<*xf64>
  %1 = "onnx.Log"(%0) : (tensor<*xf64>) -> tensor<*xf64>
  return %1 : tensor<*xf64>
}

// CHECK-LABEL: func.func @log_exp_f64(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<*xf64>
// CHECK-NOT: "onnx.Exp"
// CHECK-NOT: "onnx.Log"
// CHECK: return [[X]] : tensor<*xf64>

// -----

func.func @neg_neg_f16(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %0 = "onnx.Neg"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %1 = "onnx.Neg"(%0) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return %1 : tensor<2x3xf16>
}

// CHECK-LABEL: func.func @neg_neg_f16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf16>
// CHECK-NOT: "onnx.Neg"
// CHECK-NOT: "onnx.Neg"
// CHECK: return [[X]] : tensor<2x3xf16>

// -----

func.func @neg_neg_bf16(%x: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  %0 = "onnx.Neg"(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  %1 = "onnx.Neg"(%0) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %1 : tensor<2x3xbf16>
}

// CHECK-LABEL: func.func @neg_neg_bf16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xbf16>
// CHECK-NOT: "onnx.Neg"
// CHECK-NOT: "onnx.Neg"
// CHECK: return [[X]] : tensor<2x3xbf16>

// -----

func.func @neg_neg_f32(%x: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = "onnx.Neg"(%x) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %1 = "onnx.Neg"(%0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  return %1 : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @neg_neg_f32(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<?x3xf32>
// CHECK-NOT: "onnx.Neg"
// CHECK-NOT: "onnx.Neg"
// CHECK: return [[X]] : tensor<?x3xf32>

// -----

func.func @neg_neg_f64(%x: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "onnx.Neg"(%x) : (tensor<*xf64>) -> tensor<*xf64>
  %1 = "onnx.Neg"(%0) : (tensor<*xf64>) -> tensor<*xf64>
  return %1 : tensor<*xf64>
}

// CHECK-LABEL: func.func @neg_neg_f64(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<*xf64>
// CHECK-NOT: "onnx.Neg"
// CHECK-NOT: "onnx.Neg"
// CHECK: return [[X]] : tensor<*xf64>

// -----

func.func @neg_neg_integer(%x: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Neg"(%x) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %1 = "onnx.Neg"(%0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @neg_neg_integer(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xi32>
// CHECK-NOT: "onnx.Neg"
// CHECK: return [[X]]

// -----

func.func @neg_neg_shape_refinement(%x: tensor<?x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Neg"(%x) : (tensor<?x3xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Neg"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @neg_neg_shape_refinement(
// CHECK: [[N:%[^ ]+]] = "onnx.Neg"
// CHECK: [[R:%[^ ]+]] = "onnx.Neg"([[N]])
// CHECK: return [[R]]

// -----

func.func @neg_neg_rank_refinement(%x: tensor<*xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Neg"(%x) : (tensor<*xf32>) -> tensor<2x3xf32>
  %1 = "onnx.Neg"(%0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @neg_neg_rank_refinement(
// CHECK: [[N:%[^ ]+]] = "onnx.Neg"
// CHECK: [[R:%[^ ]+]] = "onnx.Neg"([[N]])
// CHECK: return [[R]]

// -----

func.func @mul_neg_scalar(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  %n = "onnx.Neg"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_neg_scalar(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<f32>) -> tensor<f32>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
// CHECK: return [[R]]

// -----

func.func @mul_neg_broadcast(%x: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %n = "onnx.Neg"(%x) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_neg_broadcast(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<1x3xf32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
// CHECK: return [[R]]

// -----

func.func @mul_neg_reverse(%x: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %n = "onnx.Neg"(%x) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %r = "onnx.Mul"(%c, %n) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_neg_reverse(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<1x3xf32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
// CHECK: return [[R]]

// -----

func.func @mul_neg_dynamic(%x: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<3xf32>} : () -> tensor<3xf32>
  %n = "onnx.Neg"(%x) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
  return %r : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @mul_neg_dynamic(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<?x3xf32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>
// CHECK: return [[R]]

// -----

func.func @mul_neg_integer(%x: tensor<2xi32>) -> tensor<2xi32> {
  %c = "onnx.Constant"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
  %n = "onnx.Neg"(%x) : (tensor<2xi32>) -> tensor<2xi32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %r : tensor<2xi32>
}

// CHECK-LABEL: func.func @mul_neg_integer(
// CHECK: [[N:%[^ ]+]] = "onnx.Neg"
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[N]],
// CHECK: return [[R]]

// -----

func.func @mul_neg_nonconstant(%x: tensor<2xf32>, %c: tensor<2xf32>) -> tensor<2xf32> {
  %n = "onnx.Neg"(%x) : (tensor<2xf32>) -> tensor<2xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: func.func @mul_neg_nonconstant(
// CHECK: [[N:%[^ ]+]] = "onnx.Neg"
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[N]],
// CHECK: return [[R]]

// -----

func.func @mul_neg_rank_refinement(%x: tensor<*xf32>) -> tensor<2xf32> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %n = "onnx.Neg"(%x) : (tensor<*xf32>) -> tensor<2xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: func.func @mul_neg_rank_refinement(
// CHECK: [[N:%[^ ]+]] = "onnx.Neg"
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[N]],
// CHECK: return [[R]]

// -----

func.func @mul_neg_constant_x() -> tensor<2xf32> {
  %x = "onnx.Constant"() {value = dense<3.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %n = "onnx.Neg"(%x) : (tensor<2xf32>) -> tensor<2xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: func.func @mul_neg_constant_x(
// CHECK-DAG: [[X:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK: [[N:%[^ ]+]] = "onnx.Neg"([[X]])
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[N]], [[C]])
// CHECK: return [[R]]

// -----

func.func @mul_neg_scalar_f16(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<f16>} : () -> tensor<f16>
  %n = "onnx.Neg"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  %r = "onnx.Mul"(%n, %c) : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x3xf16>
  return %r : tensor<2x3xf16>
}

// CHECK-LABEL: func.func @mul_neg_scalar_f16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf16>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<f16>) -> tensor<f16>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x3xf16>
// CHECK: return [[R]]

// -----

func.func @mul_neg_scalar_bf16(%x: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<bf16>} : () -> tensor<bf16>
  %n = "onnx.Neg"(%x) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  %r = "onnx.Mul"(%n, %c) : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
  return %r : tensor<2x3xbf16>
}

// CHECK-LABEL: func.func @mul_neg_scalar_bf16(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xbf16>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<bf16>) -> tensor<bf16>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
// CHECK: return [[R]]

// -----

func.func @mul_neg_scalar_f64(%x: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<f64>} : () -> tensor<f64>
  %n = "onnx.Neg"(%x) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %r = "onnx.Mul"(%n, %c) : (tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  return %r : tensor<2x3xf64>
}

// CHECK-LABEL: func.func @mul_neg_scalar_f64(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf64>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<f64>) -> tensor<f64>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<2x3xf64>, tensor<f64>) -> tensor<2x3xf64>
// CHECK: return [[R]]

// -----

func.func @mul_neg_nonsplat(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %c = "onnx.Constant"() {value = dense<[1.0, -2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %n = "onnx.Neg"(%x) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @mul_neg_nonsplat(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2x3xf32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00]> : tensor<3xf32>
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]]) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
// CHECK: return [[R]]

// -----

// The rewrite also fires when the Neg has other users; the Neg is then kept.
func.func @mul_neg_multi_use(%x: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %c = "onnx.Constant"() {value = dense<2.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %n = "onnx.Neg"(%x) : (tensor<2xf32>) -> tensor<2xf32>
  %r = "onnx.Mul"(%n, %c) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %n, %r : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func.func @mul_neg_multi_use(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2xf32>
// CHECK-DAG: [[C:%[^ ]+]] = onnx.Constant
// CHECK-DAG: [[N:%[^ ]+]] = "onnx.Neg"([[X]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK-DAG: [[NC:%[^ ]+]] = "onnx.Neg"([[C]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK: [[R:%[^ ]+]] = "onnx.Mul"([[X]], [[NC]])
// CHECK: return [[N]], [[R]]

// -----

// Unsafe math folds Log(Exp(x)) even where f16 Exp overflows (x > ~11.09).
func.func @log_exp_f16_overflow() -> tensor<2xf16> {
  %c = "onnx.Constant"() {value = dense<12.0> : tensor<2xf16>} : () -> tensor<2xf16>
  %0 = "onnx.Exp"(%c) : (tensor<2xf16>) -> tensor<2xf16>
  %1 = "onnx.Log"(%0) : (tensor<2xf16>) -> tensor<2xf16>
  return %1 : tensor<2xf16>
}

// CHECK-LABEL: func.func @log_exp_f16_overflow(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<1.200000e+01> : tensor<2xf16>
// CHECK-NOT: "onnx.Exp"
// CHECK-NOT: "onnx.Log"
// CHECK: return [[C]]

// -----

// Unsafe math folds Exp(Log(x)) even outside Log's domain (x < 0).
func.func @exp_log_negative() -> tensor<2xf32> {
  %c = "onnx.Constant"() {value = dense<-1.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.Log"(%c) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "onnx.Exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @exp_log_negative(
// CHECK: [[C:%[^ ]+]] = onnx.Constant dense<-1.000000e+00> : tensor<2xf32>
// CHECK-NOT: "onnx.Log"
// CHECK-NOT: "onnx.Exp"
// CHECK: return [[C]]
