// RUN: onnx-mlir-opt --canonicalize %s | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir-opt --enable-cast-data-movement-patterns=false --canonicalize %s | FileCheck %s --check-prefix=DISABLED

// -----

// DISABLED-LABEL: func.func @cast_concat(
// DISABLED:       %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1)
// DISABLED-NEXT:  %{{.*}} = "onnx.Cast"(%[[CONCAT]])
// ENABLED-LABEL:  func.func @cast_concat(
// ENABLED:        %[[CAST0:.*]] = "onnx.Cast"(%arg0)
// ENABLED:        %[[CAST1:.*]] = "onnx.Cast"(%arg1)
// ENABLED:        %[[CONCAT:.*]] = "onnx.Concat"(%[[CAST0]], %[[CAST1]])
// ENABLED-NEXT:   onnx.Return %[[CONCAT]]
func.func @cast_concat(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> tensor<5xi64> {
  %concat = "onnx.Concat"(%arg0, %arg1) {axis = 0 : si64} :
      (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
  %result = "onnx.Cast"(%concat) {to = i64} :
      (tensor<5xf32>) -> tensor<5xi64>
  "onnx.Return"(%result) : (tensor<5xi64>) -> ()
}

// -----

// DISABLED-LABEL: func.func @cast_slice(
// DISABLED:       %[[SLICE:.*]] = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4)
// DISABLED-NEXT:  %{{.*}} = "onnx.Cast"(%[[SLICE]])
// ENABLED-LABEL:  func.func @cast_slice(
// ENABLED:        %[[CAST:.*]] = "onnx.Cast"(%arg0)
// ENABLED:        %[[SLICE:.*]] = "onnx.Slice"(%[[CAST]], %arg1, %arg2, %arg3, %arg4)
// ENABLED-NEXT:   onnx.Return %[[SLICE]]
func.func @cast_slice(%arg0: tensor<4xf32>, %arg1: tensor<1xi64>,
    %arg2: tensor<1xi64>, %arg3: tensor<1xi64>, %arg4: tensor<1xi64>) ->
    tensor<2xi64> {
  %slice = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) :
      (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>,
       tensor<1xi64>) -> tensor<2xf32>
  %result = "onnx.Cast"(%slice) {to = i64} :
      (tensor<2xf32>) -> tensor<2xi64>
  "onnx.Return"(%result) : (tensor<2xi64>) -> ()
}
