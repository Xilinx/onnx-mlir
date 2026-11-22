// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

// RUN: onnx-mlir-opt --onnx-form-deferred-foldable-expressions --split-input-file %s | FileCheck %s

// Forming a consumer clones all prefix results. Dropping its unused Cast must
// not invalidate the reverse walk over the cloned expression.
func.func @dead_multiresult_prefix() -> (tensor<4xi16>, tensor<4xi16>) {
  %source = "onnx.InitializerSource"() {kind = "onnx_model", name = "bias", path = "/not/read.onnx", scope = []} : () -> tensor<4xi8>
  %prepared:2 = "onnx.DeferredFoldable"(%source) ({
  ^bb0(%s: tensor<4xi8>):
    %neg = "onnx.Neg"(%s) : (tensor<4xi8>) -> tensor<4xi8>
    %cast = "onnx.Cast"(%neg) {saturate = 1 : si64, to = i16} : (tensor<4xi8>) -> tensor<4xi16>
    onnx.Yield %neg, %cast : tensor<4xi8>, tensor<4xi16>
  }) : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi16>)
  %result = "onnx.Cast"(%prepared#0) {saturate = 1 : si64, to = i16} : (tensor<4xi8>) -> tensor<4xi16>
  return %result, %prepared#1 : tensor<4xi16>, tensor<4xi16>
}

// CHECK-LABEL: func.func @dead_multiresult_prefix
// CHECK: %[[SOURCE:.+]] = "onnx.InitializerSource"() {{.*}}name = "bias"{{.*}} : () -> tensor<4xi8>
// CHECK-NEXT: %[[PREFIX:.+]]:2 = "onnx.DeferredFoldable"(%[[SOURCE]]) ({
// CHECK-NEXT: ^bb0(%[[INPUT:.+]]: tensor<4xi8>):
// CHECK-NEXT: %[[NEG:.+]] = "onnx.Neg"(%[[INPUT]]) : (tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %[[CAST:.+]] = "onnx.Cast"(%[[NEG]]) {saturate = 1 : si64, to = i16} : (tensor<4xi8>) -> tensor<4xi16>
// CHECK-NEXT: onnx.Yield %[[NEG]], %[[CAST]] : tensor<4xi8>, tensor<4xi16>
// CHECK-NEXT: }) : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi16>)
// CHECK-NEXT: %[[RESULT:.+]] = "onnx.DeferredFoldable"(%[[SOURCE]]) ({
// CHECK-NEXT: ^bb0(%[[CAPTURE:.+]]: tensor<4xi8>):
// CHECK-NEXT: %[[RESULT_NEG:.+]] = "onnx.Neg"(%[[CAPTURE]]) : (tensor<4xi8>) -> tensor<4xi8>
// CHECK-NEXT: %[[RESULT_CAST:.+]] = "onnx.Cast"(%[[RESULT_NEG]]) {saturate = 1 : si64, to = i16} : (tensor<4xi8>) -> tensor<4xi16>
// CHECK-NEXT: onnx.Yield %[[RESULT_CAST]] : tensor<4xi16>
// CHECK-NEXT: }) : (tensor<4xi8>) -> tensor<4xi16>
// CHECK-NEXT: return %[[RESULT]], %[[PREFIX]]#1 : tensor<4xi16>, tensor<4xi16>
