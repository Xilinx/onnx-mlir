// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

func.func @add_d2s_dcr(%arg0: tensor<1x16x4x6xf32>, %arg1: tensor<1x16x4x6xf32>) -> tensor<1x4x8x12xf32> {
  %lhs = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x16x4x6xf32>) -> tensor<1x4x8x12xf32>
  %rhs = "onnx.DepthToSpace"(%arg1) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x16x4x6xf32>) -> tensor<1x4x8x12xf32>
  %add = "onnx.Add"(%lhs, %rhs) : (tensor<1x4x8x12xf32>, tensor<1x4x8x12xf32>) -> tensor<1x4x8x12xf32>
  onnx.Return %add : tensor<1x4x8x12xf32>
}

// CHECK-LABEL: func.func @add_d2s_dcr
// CHECK:         %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) : (tensor<1x16x4x6xf32>, tensor<1x16x4x6xf32>) -> tensor<1x16x4x6xf32>
// CHECK-NEXT:    %[[D2S:.*]] = "onnx.DepthToSpace"(%[[ADD]]) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x16x4x6xf32>) -> tensor<1x4x8x12xf32>
// CHECK-NEXT:    onnx.Return %[[D2S]] : tensor<1x4x8x12xf32>

// -----

func.func @add_d2s_crd(%arg0: tensor<1x27x3x3xbf16>, %arg1: tensor<1x27x3x3xbf16>) -> tensor<1x3x9x9xbf16> {
  %lhs = "onnx.DepthToSpace"(%arg0) {blocksize = 3 : si64, mode = "CRD"} : (tensor<1x27x3x3xbf16>) -> tensor<1x3x9x9xbf16>
  %rhs = "onnx.DepthToSpace"(%arg1) {blocksize = 3 : si64, mode = "CRD"} : (tensor<1x27x3x3xbf16>) -> tensor<1x3x9x9xbf16>
  %add = "onnx.Add"(%lhs, %rhs) : (tensor<1x3x9x9xbf16>, tensor<1x3x9x9xbf16>) -> tensor<1x3x9x9xbf16>
  onnx.Return %add : tensor<1x3x9x9xbf16>
}

// CHECK-LABEL: func.func @add_d2s_crd
// CHECK:         %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1) : (tensor<1x27x3x3xbf16>, tensor<1x27x3x3xbf16>) -> tensor<1x27x3x3xbf16>
// CHECK-NEXT:    %[[D2S:.*]] = "onnx.DepthToSpace"(%[[ADD]]) {blocksize = 3 : si64, mode = "CRD"} : (tensor<1x27x3x3xbf16>) -> tensor<1x3x9x9xbf16>
// CHECK-NEXT:    onnx.Return %[[D2S]] : tensor<1x3x9x9xbf16>

// -----

func.func @mul_d2s(%arg0: tensor<1x4x1x1xf32>, %arg1: tensor<1x4x1x1xf32>) -> tensor<1x1x2x2xf32> {
  %lhs = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x1x1xf32>) -> tensor<1x1x2x2xf32>
  %rhs = "onnx.DepthToSpace"(%arg1) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x1x1xf32>) -> tensor<1x1x2x2xf32>
  %mul = "onnx.Mul"(%lhs, %rhs) : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  onnx.Return %mul : tensor<1x1x2x2xf32>
}

// CHECK-LABEL: func.func @mul_d2s
// CHECK:         %[[MUL:.*]] = "onnx.Mul"(%arg0, %arg1) : (tensor<1x4x1x1xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x1x1xf32>
// CHECK-NEXT:    %[[D2S:.*]] = "onnx.DepthToSpace"(%[[MUL]]) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x4x1x1xf32>) -> tensor<1x1x2x2xf32>
// CHECK-NEXT:    onnx.Return %[[D2S]] : tensor<1x1x2x2xf32>

// -----

func.func @add_d2s_different_modes(%arg0: tensor<1x12x2x3xf32>, %arg1: tensor<1x12x2x3xf32>) -> tensor<1x3x4x6xf32> {
  %lhs = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x12x2x3xf32>) -> tensor<1x3x4x6xf32>
  %rhs = "onnx.DepthToSpace"(%arg1) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x12x2x3xf32>) -> tensor<1x3x4x6xf32>
  %add = "onnx.Add"(%lhs, %rhs) : (tensor<1x3x4x6xf32>, tensor<1x3x4x6xf32>) -> tensor<1x3x4x6xf32>
  onnx.Return %add : tensor<1x3x4x6xf32>
}

// CHECK-LABEL: func.func @add_d2s_different_modes
// CHECK:         %[[LHS:.*]] = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x12x2x3xf32>) -> tensor<1x3x4x6xf32>
// CHECK:         %[[RHS:.*]] = "onnx.DepthToSpace"(%arg1) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x12x2x3xf32>) -> tensor<1x3x4x6xf32>
// CHECK:         %[[ADD:.*]] = "onnx.Add"(%[[LHS]], %[[RHS]]) : (tensor<1x3x4x6xf32>, tensor<1x3x4x6xf32>) -> tensor<1x3x4x6xf32>
// CHECK:         onnx.Return %[[ADD]] : tensor<1x3x4x6xf32>

// -----

func.func @add_d2s_different_blocksizes(%arg0: tensor<1x8x2x2xf32>, %arg1: tensor<1x32x1x1xf32>) -> tensor<1x2x4x4xf32> {
  %lhs = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x8x2x2xf32>) -> tensor<1x2x4x4xf32>
  %rhs = "onnx.DepthToSpace"(%arg1) {blocksize = 4 : si64, mode = "DCR"} : (tensor<1x32x1x1xf32>) -> tensor<1x2x4x4xf32>
  %add = "onnx.Add"(%lhs, %rhs) : (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  onnx.Return %add : tensor<1x2x4x4xf32>
}

// CHECK-LABEL: func.func @add_d2s_different_blocksizes
// CHECK:         %[[LHS:.*]] = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR"} : (tensor<1x8x2x2xf32>) -> tensor<1x2x4x4xf32>
// CHECK:         %[[RHS:.*]] = "onnx.DepthToSpace"(%arg1) {blocksize = 4 : si64, mode = "DCR"} : (tensor<1x32x1x1xf32>) -> tensor<1x2x4x4xf32>
// CHECK:         %[[ADD:.*]] = "onnx.Add"(%[[LHS]], %[[RHS]]) : (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
// CHECK:         onnx.Return %[[ADD]] : tensor<1x2x4x4xf32>
