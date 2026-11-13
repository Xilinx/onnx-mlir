// RUN: onnx-mlir-opt --decompose-onnx=enable-split-to-slice --shape-inference %s -split-input-file | FileCheck %s

// A symbolic dimension is not a negative end index. Only the split axis may
// appear in Slice's bounds, including when unknown dimensions surround it.
func.func @explicit_split_preserves_other_dimensions(%arg0: tensor<?x6x?xf32>) -> (tensor<?x2x?xf32>, tensor<?x4x?xf32>) {
  %sizes = onnx.Constant dense<[2, 4]> : tensor<2xi64>
  %split:2 = "onnx.Split"(%arg0, %sizes) {axis = 1 : si64} : (tensor<?x6x?xf32>, tensor<2xi64>) -> (tensor<?x2x?xf32>, tensor<?x4x?xf32>)
  return %split#0, %split#1 : tensor<?x2x?xf32>, tensor<?x4x?xf32>
}

// CHECK-LABEL: func.func @explicit_split_preserves_other_dimensions
// CHECK-SAME: ([[INPUT:%.+]]: tensor<?x6x?xf32>)
// CHECK-DAG: [[ZERO:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG: [[ONE:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG: [[TWO:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG: [[SIX:%.+]] = onnx.Constant dense<6> : tensor<1xi64>
// CHECK: [[FIRST:%.+]] = "onnx.Slice"([[INPUT]], [[ZERO]], [[TWO]], [[ONE]], [[ONE]]) : (tensor<?x6x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x2x?xf32>
// CHECK: [[SECOND:%.+]] = "onnx.Slice"([[INPUT]], [[TWO]], [[SIX]], [[ONE]], [[ONE]]) : (tensor<?x6x?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x4x?xf32>
// CHECK: return [[FIRST]], [[SECOND]]

// -----

// Default split sizes and a negative axis must also preserve the sequence
// dimension. Using the dynamic-dimension sentinel as an end index made it zero.
func.func @default_split_negative_axis(%arg0: tensor<3x1x?x8xf32>) -> (tensor<3x1x?x4xf32>, tensor<3x1x?x4xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %split:2 = "onnx.Split"(%arg0, %none) {axis = -1 : si64} : (tensor<3x1x?x8xf32>, none) -> (tensor<3x1x?x4xf32>, tensor<3x1x?x4xf32>)
  return %split#0, %split#1 : tensor<3x1x?x4xf32>, tensor<3x1x?x4xf32>
}

// CHECK-LABEL: func.func @default_split_negative_axis
// CHECK-SAME: ([[INPUT:%.+]]: tensor<3x1x?x8xf32>)
// CHECK-DAG: [[ZERO:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG: [[ONE:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG: [[AXIS:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG: [[FOUR:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG: [[EIGHT:%.+]] = onnx.Constant dense<8> : tensor<1xi64>
// CHECK: [[FIRST:%.+]] = "onnx.Slice"([[INPUT]], [[ZERO]], [[FOUR]], [[AXIS]], [[ONE]]) : (tensor<3x1x?x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x1x?x4xf32>
// CHECK: [[SECOND:%.+]] = "onnx.Slice"([[INPUT]], [[FOUR]], [[EIGHT]], [[AXIS]], [[ONE]]) : (tensor<3x1x?x8xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3x1x?x4xf32>
// CHECK: return [[FIRST]], [[SECOND]]
