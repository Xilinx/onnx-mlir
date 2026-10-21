// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

module {
  func.func @wrong_element_type() {
    // expected-error @+1 {{source data_type disagrees with result element type}}
    %0 = "onnx.InitializerSource"() {source = #onnx.initializer_source<{data_type = 1 : i64, length = 4 : i64, name = "W", offset = 0 : i64, path = "/not/read", scope = ""}>} : () -> tensor<1xi32>
    return
  }
}

// -----

module {
  func.func @wrong_length() {
    // expected-error @+1 {{source length disagrees with tensor payload size}}
    %0 = "onnx.InitializerSource"() {source = #onnx.initializer_source<{data_type = 1 : i64, length = 8 : i64, name = "W", offset = 0 : i64, path = "/not/read", scope = ""}>} : () -> tensor<1xf32>
    return
  }
}
