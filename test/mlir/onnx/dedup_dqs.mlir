// RUN: onnx-mlir-opt -dedup-dqs %s -split-input-file | FileCheck %s

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
// CHECK-COUNT-1: [[DQ:%[0-9]+]] = "onnx.DequantizeLinear"
// CHECK-NEXT: return [[DQ]], [[DQ]], [[DQ]]
