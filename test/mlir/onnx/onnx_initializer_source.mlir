// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: onnx-mlir-opt %s --onnx-materialize-initializer-sources="skip-all" | FileCheck %s --check-prefix=ALL
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-names=W0,W1" | FileCheck %s --check-prefix=NAMED
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-all force-names=W2" | FileCheck %s --check-prefix=FORCED
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-names=MISSING,W0,W1,W2" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: onnx-mlir-opt %s --onnx-materialize-initializer-sources="skip-all force-names=MISSING" 2>&1 | FileCheck %s --check-prefix=UNKNOWN-FORCE

module {
  func.func @sources() {
    %0 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "W0", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
    %1 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "W1", offset = 4 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
    %2 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "W2", offset = 0 : i64, path = "@DATA@", scope = []} : () -> tensor<1xf32>
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
// FORCED: name = "W0"
// FORCED: name = "W1"
// FORCED-NOT: name = "W2"
// FORCED: onnx.Constant
// UNKNOWN: unknown initializer source in skip-names: MISSING
// UNKNOWN-FORCE: unknown initializer source in force-names: MISSING
