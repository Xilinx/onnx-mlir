// RUN: onnx-mlir-opt --canonicalize %s -split-input-file | FileCheck %s


// A rank-1 singleton index preserves the data rank and selects the only
// element of the leading singleton axis.
func.func @identity_axis0(%data: tensor<1x1x2560xf32>) -> tensor<1x1x2560xf32> {
  %indices = onnx.Constant dense<0> : tensor<1xi64>
  %gather = "onnx.Gather"(%data, %indices) {axis = 0 : si64}
      : (tensor<1x1x2560xf32>, tensor<1xi64>) -> tensor<1x1x2560xf32>
  return %gather : tensor<1x1x2560xf32>
}

// CHECK-LABEL:  func.func @identity_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2560xf32>) -> tensor<1x1x2560xf32> {
// CHECK-NOT:       "onnx.Gather"
// CHECK:           return [[PARAM_0_]] : tensor<1x1x2560xf32>
// CHECK:         }

// -----

// Negative axes and -1 indices select the same singleton element.
func.func @identity_negative_axis(%data: tensor<2x1x3xf32>) -> tensor<2x1x3xf32> {
  %indices = onnx.Constant dense<-1> : tensor<1xi64>
  %gather = "onnx.Gather"(%data, %indices) {axis = -2 : si64}
      : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
  return %gather : tensor<2x1x3xf32>
}

// CHECK-LABEL:  func.func @identity_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x3xf32>) -> tensor<2x1x3xf32> {
// CHECK-NOT:       "onnx.Gather"
// CHECK:           return [[PARAM_0_]] : tensor<2x1x3xf32>
// CHECK:         }

// -----

// A scalar index removes the selected singleton axis.
func.func @scalar_singleton_axis(%data: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
  %indices = onnx.Constant dense<0> : tensor<i64>
  %gather = "onnx.Gather"(%data, %indices) {axis = 1 : si64}
      : (tensor<2x1x3xf32>, tensor<i64>) -> tensor<2x3xf32>
  return %gather : tensor<2x3xf32>
}

// CHECK-LABEL:  func.func @scalar_singleton_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
// CHECK-NOT:       "onnx.Gather"
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x1x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x3xf32>
// CHECK:         }

// -----

// Scalar -1 follows the same Squeeze path.
func.func @scalar_negative_index(%data: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
  %indices = onnx.Constant dense<-1> : tensor<i64>
  %gather = "onnx.Gather"(%data, %indices) {axis = 1 : si64}
      : (tensor<2x1x3xf32>, tensor<i64>) -> tensor<2x3xf32>
  return %gather : tensor<2x3xf32>
}

// CHECK-LABEL:  func.func @scalar_negative_index
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
// CHECK-NOT:       "onnx.Gather"
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<2x1x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x3xf32>
// CHECK:         }

// -----

// Repeated indices replicate a singleton slice, so this is not an identity.
func.func @repeated_indices_not_folded(%data: tensor<2x1x3xf32>) -> tensor<2x2x3xf32> {
  %indices = onnx.Constant dense<[0, 0]> : tensor<2xi64>
  %gather = "onnx.Gather"(%data, %indices) {axis = 1 : si64}
      : (tensor<2x1x3xf32>, tensor<2xi64>) -> tensor<2x2x3xf32>
  return %gather : tensor<2x2x3xf32>
}

// CHECK-LABEL:  func.func @repeated_indices_not_folded
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x3xf32>) -> tensor<2x2x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Gather"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<2x1x3xf32>, tensor<2xi64>) -> tensor<2x2x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x2x3xf32>
// CHECK:         }

// -----

func.func @rank2_indices_not_folded(%data: tensor<2x1x3xf32>) -> tensor<2x1x1x3xf32> {
  %indices = onnx.Constant dense<0> : tensor<1x1xi64>
  %gather = "onnx.Gather"(%data, %indices) {axis = 1 : si64}
      : (tensor<2x1x3xf32>, tensor<1x1xi64>) -> tensor<2x1x1x3xf32>
  return %gather : tensor<2x1x1x3xf32>
}

// CHECK-LABEL:  func.func @rank2_indices_not_folded
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x3xf32>) -> tensor<2x1x1x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1x1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Gather"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<2x1x3xf32>, tensor<1x1xi64>) -> tensor<2x1x1x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x1x1x3xf32>
// CHECK:         }

// -----

// A runtime index cannot be proven to select the singleton value.
func.func @dynamic_index_not_folded(%data: tensor<2x1x3xf32>, %indices: tensor<1xi64>) -> tensor<2x1x3xf32> {
  %gather = "onnx.Gather"(%data, %indices) {axis = 1 : si64}
      : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
  return %gather : tensor<2x1x3xf32>
}

// CHECK-LABEL:  func.func @dynamic_index_not_folded
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x3xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<2x1x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Gather"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<2x1x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x1x3xf32>
// CHECK:         }

// -----

// Selecting index zero from a non-singleton axis is not an identity.
func.func @non_singleton_axis_not_folded(%data: tensor<2x2x3xf32>) -> tensor<2x1x3xf32> {
  %indices = onnx.Constant dense<0> : tensor<1xi64>
  %gather = "onnx.Gather"(%data, %indices) {axis = 1 : si64}
      : (tensor<2x2x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
  return %gather : tensor<2x1x3xf32>
}

// CHECK-LABEL:  func.func @non_singleton_axis_not_folded
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x2x3xf32>) -> tensor<2x1x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Gather"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<2x2x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x1x3xf32>
// CHECK:         }

