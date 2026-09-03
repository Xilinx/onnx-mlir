// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

func.func @gather_elements_ui16(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]> : tensor<3x4xui16>} : () -> tensor<3x4xui16>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x4xf32>, tensor<3x4xui16>) -> tensor<3x4xf32>
  "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
// CHECK-LABEL: func.func @gather_elements_ui16
// CHECK: "onnx.GatherElements"({{.*}}) {axis = 1 : si64} : (tensor<3x4xf32>, tensor<3x4xui16>) -> tensor<3x4xf32>
}

// -----

func.func @gather_elements_ui32(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1, 2], [2, 1, 0]]> : tensor<2x3xui32>} : () -> tensor<2x3xui32>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
  "onnx.Return"(%0) : (tensor<2x3xf32>) -> ()
// CHECK-LABEL: func.func @gather_elements_ui32
// CHECK: "onnx.GatherElements"({{.*}}) {axis = 1 : si64} : (tensor<2x3xf32>, tensor<2x3xui32>) -> tensor<2x3xf32>
}

// -----

func.func @gather_nd_ui16(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xui16>) -> tensor<2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xui16>) -> tensor<2xf32>
  "onnx.Return"(%0) : (tensor<2xf32>) -> ()
// CHECK-LABEL: func.func @gather_nd_ui16
// CHECK: "onnx.GatherND"({{.*}}) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xui16>) -> tensor<2xf32>
}

// -----

func.func @gather_ui16(%arg0: tensor<3x2xf32>, %arg1: tensor<2x2xui16>) -> tensor<2x2x2xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xui16>) -> tensor<2x2x2xf32>
  "onnx.Return"(%0) : (tensor<2x2x2xf32>) -> ()
// CHECK-LABEL: func.func @gather_ui16
// CHECK: "onnx.Gather"({{.*}}) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xui16>) -> tensor<2x2x2xf32>
}

// -----

func.func @scatter_elements_ui16(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %indices = "onnx.Constant"() {value = dense<[0, 2, 1]> : tensor<3xui16>} : () -> tensor<3xui16>
  %0 = "onnx.ScatterElements"(%arg0, %indices, %arg1) {axis = 0 : si64, reduction = "none"} : (tensor<3xf32>, tensor<3xui16>, tensor<3xf32>) -> tensor<3xf32>
  "onnx.Return"(%0) : (tensor<3xf32>) -> ()
// CHECK-LABEL: func.func @scatter_elements_ui16
// CHECK: "onnx.ScatterElements"({{.*}}) {axis = 0 : si64, reduction = "none"} : (tensor<3xf32>, tensor<3xui16>, tensor<3xf32>) -> tensor<3xf32>
}

// -----

func.func @scatter_nd_ui16(%arg0: tensor<4x4xf32>, %arg1: tensor<2xf32>) -> tensor<4x4xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 0], [1, 1]]> : tensor<2x2xui16>} : () -> tensor<2x2xui16>
  %0 = "onnx.ScatterND"(%arg0, %indices, %arg1) {reduction = "none"} : (tensor<4x4xf32>, tensor<2x2xui16>, tensor<2xf32>) -> tensor<4x4xf32>
  "onnx.Return"(%0) : (tensor<4x4xf32>) -> ()
// CHECK-LABEL: func.func @scatter_nd_ui16
// CHECK: "onnx.ScatterND"({{.*}}) {reduction = "none"} : (tensor<4x4xf32>, tensor<2x2xui16>, tensor<2xf32>) -> tensor<4x4xf32>
}

// -----

func.func @scatter_nd_ui32(%arg0: tensor<2x3xf32>, %arg1: tensor<2xf32>) -> tensor<2x3xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xui32>} : () -> tensor<2x2xui32>
  %0 = "onnx.ScatterND"(%arg0, %indices, %arg1) {reduction = "none"} : (tensor<2x3xf32>, tensor<2x2xui32>, tensor<2xf32>) -> tensor<2x3xf32>
  "onnx.Return"(%0) : (tensor<2x3xf32>) -> ()
// CHECK-LABEL: func.func @scatter_nd_ui32
// CHECK: "onnx.ScatterND"({{.*}}) {reduction = "none"} : (tensor<2x3xf32>, tensor<2x2xui32>, tensor<2xf32>) -> tensor<2x3xf32>
}

// -----

// GatherElements inherits ui16 indices from TopK; no ui16 literal on the GatherElements op.
func.func @gather_elements_from_topk_ui16(%arg0: tensor<3x4xf32>) -> tensor<*xf32> {
  %k = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %values, %indices = "onnx.TopK"(%arg0, %k) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xui16>)
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x4xf32>, tensor<*xui16>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func.func @gather_elements_from_topk_ui16
// CHECK-DAG: "onnx.TopK"({{.*}}) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<3x2xf32>, tensor<3x2xui16>)
// CHECK: "onnx.GatherElements"({{.*}}) {axis = 1 : si64} : (tensor<3x4xf32>, tensor<3x2xui16>) -> tensor<3x2xf32>
}

// -----

// Gather: ranked ui16 indices argument, unranked result type on the op.
func.func @gather_ui16_unranked(%arg0: tensor<3x2xf32>, %arg1: tensor<2x2xui16>) -> tensor<*xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xui16>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func.func @gather_ui16_unranked
// CHECK: "onnx.Gather"({{.*}}) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xui16>) -> tensor<2x2x2xf32>
}

// -----

// ScatterElements inherits ui16 indices from TopK.
func.func @scatter_elements_from_topk_ui16(%arg0: tensor<3x4xf32>, %updates: tensor<3x2xf32>) -> tensor<*xf32> {
  %k = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  %values, %indices = "onnx.TopK"(%arg0, %k) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xui16>)
  %0 = "onnx.ScatterElements"(%arg0, %indices, %updates) {axis = 1 : si64, reduction = "none"} : (tensor<3x4xf32>, tensor<*xui16>, tensor<3x2xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func.func @scatter_elements_from_topk_ui16
// CHECK-DAG: "onnx.TopK"({{.*}}) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<3x2xf32>, tensor<3x2xui16>)
// CHECK: "onnx.ScatterElements"({{.*}}) {axis = 1 : si64, reduction = "none"} : (tensor<3x4xf32>, tensor<3x2xui16>, tensor<3x2xf32>) -> tensor<3x4xf32>
}
