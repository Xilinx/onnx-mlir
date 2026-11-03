// SPDX-License-Identifier: Apache-2.0
// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its
// affiliates
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-all" | FileCheck %s --check-prefix=ALL
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-names=W0,W1,RandomShapeSeed" | FileCheck %s --check-prefix=NAMED
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-all force-names=W2" | FileCheck %s --check-prefix=FORCED
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-names=MISSING,W0,W1,W2,RandomShapeSeed" 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: %python -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text().replace('@DATA@', p.resolve().as_posix()))" %s | onnx-mlir-opt --onnx-materialize-initializer-sources="skip-all force-names=MISSING" 2>&1 | FileCheck %s --check-prefix=UNKNOWN-FORCE

module {
  func.func @sources() {
    %0 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "W0", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
    %1 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "W1", offset = 4 : i64, path = "/not/read", scope = []} : () -> tensor<1xf32>
    %2 = "onnx.InitializerSource"() {kind = "external_data", length = 4 : i64, name = "W2", offset = 0 : i64, path = "@DATA@", scope = []} : () -> tensor<1xf32>
    return
  }
  func.func @loop_structural_operand() -> tensor<2xi64> {
    %trip = onnx.Constant dense<1> : tensor<i64>
    %cond = onnx.Constant dense<true> : tensor<i1>
    %shape = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "LoopShape", offset = 0 : i64, path = "@DATA@", scope = []} : () -> tensor<2xi64>
    %result = "onnx.Loop"(%trip, %cond, %shape) ({
    ^bb0(%iteration: tensor<i64>, %cond_in: tensor<i1>, %shape_in: tensor<2xi64>):
      %data = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
      %reshaped = "onnx.Reshape"(%data, %shape_in) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
      onnx.Yield %cond_in, %shape_in : tensor<i1>, tensor<2xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<2xi64>) -> tensor<2xi64>
    return %result : tensor<2xi64>
  }
  func.func @derived_structural_operand() -> tensor<?x?xf32> {
    %shape = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "DerivedShape", offset = 0 : i64, path = "@DATA@", scope = []} : () -> tensor<2xi64>
    %zero = onnx.Constant dense<0> : tensor<2xi64>
    %derived_shape = "onnx.Add"(%shape, %zero) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %data = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %reshaped = "onnx.Reshape"(%data, %derived_shape) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %reshaped : tensor<?x?xf32>
  }
  func.func @mixed_runtime_structural_operand(%runtime: tensor<2xi64>) -> tensor<?x?xf32> {
    %shape = "onnx.InitializerSource"() {kind = "external_data", length = 16 : i64, name = "MixedShape", offset = 0 : i64, path = "@DATA@", scope = []} : () -> tensor<2xi64>
    %mixed_shape = "onnx.Add"(%shape, %runtime) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %data = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %reshaped = "onnx.Reshape"(%data, %mixed_shape) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %reshaped : tensor<?x?xf32>
  }
  func.func @nondeterministic_structural_operand() -> tensor<?x?xf32> {
    %seed_data = "onnx.InitializerSource"() {kind = "external_data", length = 8 : i64, name = "RandomShapeSeed", offset = 0 : i64, path = "/not/read", scope = []} : () -> tensor<2xf32>
    %random = "onnx.RandomNormalLike"(%seed_data) {dtype = 1 : si64, mean = 0.0 : f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<2xf32>) -> tensor<2xf32>
    %shape = "onnx.Cast"(%random) {to = i64} : (tensor<2xf32>) -> tensor<2xi64>
    %data = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %reshaped = "onnx.Reshape"(%data, %shape) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %reshaped : tensor<?x?xf32>
  }
}

// ALL-LABEL: func.func @sources
// ALL-COUNT-3: "onnx.InitializerSource"
// ALL-NOT: "onnx.Constant"
// ALL-LABEL: func.func @loop_structural_operand
// ALL-NOT: name = "LoopShape"
// ALL: onnx.Constant
// ALL-LABEL: func.func @derived_structural_operand
// ALL-NOT: name = "DerivedShape"
// ALL: onnx.Constant
// ALL: onnx.Add
// ALL: onnx.Reshape
// ALL-LABEL: func.func @mixed_runtime_structural_operand
// ALL: name = "MixedShape"
// ALL: onnx.Add
// ALL-LABEL: func.func @nondeterministic_structural_operand
// ALL: name = "RandomShapeSeed"
// ALL: "onnx.RandomNormalLike"
// ALL: "onnx.Cast"
// ALL: "onnx.Reshape"
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
