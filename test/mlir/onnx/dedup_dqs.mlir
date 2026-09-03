// RUN: onnx-mlir-opt --split-input-file -dedup-dqs %s -split-input-file | FileCheck %s

func.func @duplicate_dqs() -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = onnx.Constant dense<127> : tensor<64xi8>
  %2 = onnx.Constant dense<0.00787401571> : tensor<f32>
  %3 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "a"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %4 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "b"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %5 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "c"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  return %3, %4, %5 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
}

// CHECK-LABEL: @duplicate_dqs
// CHECK: [[DQ:%[0-9]+]] = "onnx.DequantizeLinear"
// CHECK-SAME: onnx_node_name = "a"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NEXT: return [[DQ]], [[DQ]], [[DQ]]

// -----

// Second DQ is output -> That will be kept
func.func @dedup_prefer_output_dq(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = onnx.Constant dense<127> : tensor<64xi8>
  %2 = onnx.Constant dense<0.00787401571> : tensor<f32>
  %3 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "a"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %5 = "onnx.Add"(%3, %arg0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %4 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "b"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  return %5, %4 : tensor<64xf32>, tensor<64xf32>
}

// CHECK-LABEL: @dedup_prefer_output_dq
// CHECK: [[DQ:%.*]] = "onnx.DequantizeLinear"
// CHECK-SAME: onnx_node_name = "b"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: [[ADD:%.*]] = "onnx.Add"([[DQ]],
// CHECK: return [[ADD]], [[DQ]]

// -----

// Earlier DQ is an output -> That will be kept
func.func @dedup_keep_output_when_first(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = onnx.Constant dense<127> : tensor<64xi8>
  %2 = onnx.Constant dense<0.00787401571> : tensor<f32>
  %3 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "a"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %4 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "b"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %5 = "onnx.Add"(%4, %arg0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %3, %5 : tensor<64xf32>, tensor<64xf32>
}

// CHECK-LABEL: @dedup_keep_output_when_first
// CHECK: [[DQ:%.*]] = "onnx.DequantizeLinear"
// CHECK-SAME: onnx_node_name = "a"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: [[ADD:%.*]] = "onnx.Add"([[DQ]],
// CHECK: return [[DQ]], [[ADD]]

// -----

// Neither DQ is an output -> First one will be kept
func.func @dedup_no_output_dqs(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = onnx.Constant dense<127> : tensor<64xi8>
  %2 = onnx.Constant dense<0.00787401571> : tensor<f32>
  %3 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "a"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %4 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "b"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %5 = "onnx.Add"(%3, %arg0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %6 = "onnx.Add"(%4, %arg0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %5, %6 : tensor<64xf32>, tensor<64xf32>
}

// CHECK-LABEL: @dedup_no_output_dqs
// CHECK: [[DQ:%.*]] = "onnx.DequantizeLinear"
// CHECK: onnx_node_name = "a"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: [[ADD1:%.*]] = "onnx.Add"([[DQ]],
// CHECK: [[ADD2:%.*]] = "onnx.Add"([[DQ]],
// CHECK: return [[ADD1]], [[ADD2]]

// -----

// Three duplicate DQs: first two are non-outputs, third is an output.
// Tests the set update logic: first dedup replaces "b" with "a" (else-branch),
// second dedup replaces "a" with "c" (if-branch, set erase+insert).
func.func @dedup_three_mixed_output(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
  %0 = onnx.Constant dense<0> : tensor<i8>
  %1 = onnx.Constant dense<127> : tensor<64xi8>
  %2 = onnx.Constant dense<0.00787401571> : tensor<f32>
  %3 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "a"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %4 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "b"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %5 = "onnx.DequantizeLinear"(%1, %2, %0) {axis = 1 : si64, block_size = 0 : si64, onnx_node_name = "c"} : (tensor<64xi8>, tensor<f32>, tensor<i8>) -> tensor<64xf32>
  %6 = "onnx.Add"(%3, %arg0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %7 = "onnx.Add"(%4, %arg0) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %6, %7, %5 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
}

// CHECK-LABEL: @dedup_three_mixed_output
// CHECK: [[DQ:%.*]] = "onnx.DequantizeLinear"
// CHECK-SAME: onnx_node_name = "c"
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK: [[ADD1:%.*]] = "onnx.Add"([[DQ]],
// CHECK: [[ADD2:%.*]] = "onnx.Add"([[DQ]],
// CHECK: return [[ADD1]], [[ADD2]], [[DQ]]
