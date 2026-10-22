// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: onnx-mlir-opt %s --onnx-materialize-initializer-sources="skip-all" | FileCheck %s --check-prefix=ALL
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-names=W0,W1" | FileCheck %s --check-prefix=NAMED
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('/not/read', p.resolve().as_posix()).replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-names=MISSING" 2>&1 | FileCheck %s --check-prefix=UNKNOWN

module {
  func.func @sources() {
    %0 = "onnx.InitializerSource"() {data_type = 1 : i64, kind = "external_data", length = 4 : i64, name = "W0", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
    %1 = "onnx.InitializerSource"() {data_type = 1 : i64, kind = "external_data", length = 4 : i64, name = "W1", offset = 4 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
    %2 = "onnx.InitializerSource"() {data_type = 1 : i64, kind = "external_data", length = 4 : i64, name = "W2", offset = 0 : i64, path = "@DATA@", scope = []} : () -> tensor<1xf32>
    return
  }
}

// ALL-COUNT-3: "onnx.InitializerSource"
// ALL-NOT: "onnx.Constant"
// NAMED: name = "W0"
// NAMED: name = "W1"
// NAMED-NOT: "onnx.InitializerSource"
// NAMED-NOT: name = "W2"
// NAMED: onnx.Constant
// UNKNOWN: unknown initializer source in skip-names: MISSING
