// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: onnx-mlir-opt %s --onnx-materialize-initializer-sources="skip-all" | FileCheck %s --check-prefix=ALL
// RUN: onnx-mlir-opt %s --onnx-materialize-initializer-sources="skip-names=W0,W1" | FileCheck %s --check-prefix=NAMED
// RUN: not onnx-mlir-opt %s --onnx-materialize-initializer-sources="skip-names=MISSING" 2>&1 | FileCheck %s --check-prefix=UNKNOWN

module {
  func.func @sources() {
    %0 = "onnx.InitializerSource"() {source = #onnx.initializer_source<{data_type = 1 : i64, length = 4 : i64, name = "W0", offset = 0 : i64, path = "/not/read", scope = ""}>} : () -> tensor<1xf32>
    %1 = "onnx.InitializerSource"() {source = #onnx.initializer_source<{data_type = 1 : i64, length = 4 : i64, name = "W1", offset = 4 : i64, path = "/not/read", scope = ""}>} : () -> tensor<1xf32>
    return
  }
}

// ALL-COUNT-2: "onnx.InitializerSource"
// NAMED-COUNT-2: "onnx.InitializerSource"
// UNKNOWN: unknown initializer source in skip-names: MISSING
