// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s --check-prefix=OFF
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-depthtospace-decompose --recompose-onnx=enable-depthtospace-decompose --canonicalize %s -split-input-file | FileCheck %s --check-prefix=ON

// -----

// Reshape/Transpose/Reshape (CRD) as input.
func.func @test_rtr_to_dts(%arg0: tensor<1x128x540x960xf32>) -> tensor<1x32x1080x1920xf32> {
  %0 = onnx.Constant dense<[-1, 32, 2, 2, 540, 960]> : tensor<6xi64>
  %1 = onnx.Constant dense<[-1, 32, 1080, 1920]> : tensor<4xi64>
  %2 = "onnx.Reshape"(%arg0, %0) {allowzero = 0 : si64} : (tensor<1x128x540x960xf32>, tensor<6xi64>) -> tensor<1x32x2x2x540x960xf32>
  %3 = "onnx.Transpose"(%2) {perm = [0, 1, 4, 2, 5, 3]} : (tensor<1x32x2x2x540x960xf32>) -> tensor<1x32x540x2x960x2xf32>
  %4 = "onnx.Reshape"(%3, %1) {allowzero = 0 : si64} : (tensor<1x32x540x2x960x2xf32>, tensor<4xi64>) -> tensor<1x32x1080x1920xf32>
  return %4 : tensor<1x32x1080x1920xf32>
}
// OFF-LABEL:  func.func @test_rtr_to_dts
// OFF:          "onnx.DepthToSpace"(%{{.*}}) {blocksize = 2 : si64, mode = "CRD"}
// OFF:          return

// ON-LABEL:   func.func @test_rtr_to_dts
// ON-NOT:       onnx.DepthToSpace
// ON:           "onnx.Reshape"
// ON:           "onnx.Transpose"
// ON:           "onnx.Reshape"
// ON:           return

// -----

// onnx.DepthToSpace (DCR) as input.
func.func @test_dts_to_rtr(%arg0: tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32> {
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x12x4x5xf32>) -> tensor<1x3x8x10xf32>
  onnx.Return %0 : tensor<1x3x8x10xf32>
}
// OFF-LABEL:  func.func @test_dts_to_rtr
// OFF:          "onnx.DepthToSpace"(%{{.*}}) {blocksize = 2 : si64, mode = "DCR"}
// OFF:          onnx.Return

// ON-LABEL:   func.func @test_dts_to_rtr
// ON-NOT:       onnx.DepthToSpace
// ON:           "onnx.Reshape"
// ON:           "onnx.Transpose"
// ON:           "onnx.Reshape"
// ON:           onnx.Return
