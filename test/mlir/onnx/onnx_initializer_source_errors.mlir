// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

func.func @unsupported_element_type() {
  // expected-error @+1 {{has unsupported source element type 'complex<f32>'}}
  %0 = "onnx.InitializerSource"() {kind = "external_data", length = 8 : i64, name = "W", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1xcomplex<f32>>
  return
}

// -----

func.func @wrong_length() {
  // expected-error @+1 {{source length disagrees with tensor payload size}}
  %0 = "onnx.InitializerSource"() {kind = "external_data", length = 8 : i64, name = "W", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
  return
}

// -----

func.func @unknown_source_kind() {
  // expected-error @+1 {{has unknown source kind unknown}}
  %0 = "onnx.InitializerSource"() {kind = "unknown", name = "W", path = "/not/read", scope = []} : () -> tensor<1xf32>
  return
}

// -----

func.func @model_source_with_range() {
  // expected-error @+1 {{onnx_model source must not contain offset/length}}
  %0 = "onnx.InitializerSource"() {kind = "onnx_model", length = 4 : i64, name = "W", offset = 0 : i64, path = "/model.onnx", scope = []} : () -> tensor<1xf32>
  return
}

func.func @empty_name() {
  // expected-error @+1 {{requires a nonempty initializer name}}
  %0 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
  return
}
