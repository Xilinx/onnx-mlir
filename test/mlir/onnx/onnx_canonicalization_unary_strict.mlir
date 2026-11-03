// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

func.func @exp_log_strict(%x: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Log"(%x) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "onnx.Exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @exp_log_strict(
// CHECK: [[I:%[^ ]+]] = "onnx.Log"
// CHECK: [[R:%[^ ]+]] = "onnx.Exp"([[I]])
// CHECK: return [[R]]

// -----

func.func @log_exp_strict(%x: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Exp"(%x) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "onnx.Log"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @log_exp_strict(
// CHECK: [[I:%[^ ]+]] = "onnx.Exp"
// CHECK: [[R:%[^ ]+]] = "onnx.Log"([[I]])
// CHECK: return [[R]]

// -----

func.func @log_exp_f16_overflow_strict() -> tensor<2xf16> {
  %c = "onnx.Constant"() {value = dense<12.0> : tensor<2xf16>} : () -> tensor<2xf16>
  %0 = "onnx.Exp"(%c) : (tensor<2xf16>) -> tensor<2xf16>
  %1 = "onnx.Log"(%0) : (tensor<2xf16>) -> tensor<2xf16>
  return %1 : tensor<2xf16>
}

// CHECK-LABEL: func.func @log_exp_f16_overflow_strict(
// CHECK: [[I:%[^ ]+]] = "onnx.Exp"
// CHECK: [[R:%[^ ]+]] = "onnx.Log"([[I]])
// CHECK: return [[R]]

// -----

func.func @neg_neg_strict(%x: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "onnx.Neg"(%x) : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "onnx.Neg"(%0) : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @neg_neg_strict(
// CHECK-SAME: [[X:%[^ ]+]]: tensor<2xf32>
// CHECK-NOT: "onnx.Neg"
// CHECK: return [[X]]
