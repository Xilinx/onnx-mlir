// RUN: onnx-mlir-opt --onnx-hybrid-transform="shape-inference=false canonicalization=false constant-propagation=false enable-reducel2-decompose=false enable-reducel2-recompositions=true" %s -split-input-file | FileCheck %s --check-prefix=RECOMPOSE
// RUN: onnx-mlir-opt --onnx-hybrid-transform="shape-inference=false canonicalization=false constant-propagation=false enable-reducel2-decompose=true enable-reducel2-recompositions=true" %s -split-input-file | FileCheck %s --check-prefix=DECOMPOSE

func.func @test_full_reducel2_chain(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %square = "onnx.Mul"(%arg0, %arg0) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %sum = "onnx.ReduceSum"(%square, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  %result = "onnx.Sqrt"(%sum) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// RECOMPOSE-LABEL:  func.func @test_full_reducel2_chain
// RECOMPOSE-NOT:       "onnx.ReduceSumSquare"
// RECOMPOSE:           [[L2_:%.+]] = "onnx.ReduceL2"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
// RECOMPOSE:           onnx.Return [[L2_]] : tensor<?x?xf32>
// DECOMPOSE-LABEL:  func.func @test_full_reducel2_chain
// DECOMPOSE-NOT:       "onnx.ReduceL2"
// DECOMPOSE-NOT:       "onnx.ReduceSumSquare"
// DECOMPOSE:           [[SQUARE_:%.+]] = "onnx.Mul"(%arg0, %arg0)
// DECOMPOSE:           [[SUM_:%.+]] = "onnx.ReduceSum"([[SQUARE_]], %arg1)
// DECOMPOSE:           [[RESULT_:%.+]] = "onnx.Sqrt"([[SUM_]])
// DECOMPOSE:           onnx.Return [[RESULT_]] : tensor<?x?xf32>
}

// -----

func.func @test_standalone_reducesumsquare(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %result = "onnx.ReduceSumSquare"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// RECOMPOSE-LABEL:  func.func @test_standalone_reducesumsquare
// RECOMPOSE-NOT:       "onnx.ReduceSumSquare"
// RECOMPOSE:           [[SQUARE_:%.+]] = "onnx.Mul"(%arg0, %arg0)
// RECOMPOSE:           [[SUM_:%.+]] = "onnx.ReduceSum"([[SQUARE_]], %arg1)
// RECOMPOSE:           onnx.Return [[SUM_]] : tensor<?x?xf32>
// DECOMPOSE-LABEL:  func.func @test_standalone_reducesumsquare
// DECOMPOSE-NOT:       "onnx.ReduceSumSquare"
// DECOMPOSE:           [[SQUARE_:%.+]] = "onnx.Mul"(%arg0, %arg0)
// DECOMPOSE:           [[SUM_:%.+]] = "onnx.ReduceSum"([[SQUARE_]], %arg1)
// DECOMPOSE:           onnx.Return [[SUM_]] : tensor<?x?xf32>
}

// -----

func.func @test_reducel2_decomposition_precedence(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32> {
  %result = "onnx.ReduceL2"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
  onnx.Return %result : tensor<?x?xf32>
// RECOMPOSE-LABEL:  func.func @test_reducel2_decomposition_precedence
// RECOMPOSE-NOT:       "onnx.ReduceSumSquare"
// RECOMPOSE:           [[L2_:%.+]] = "onnx.ReduceL2"(%arg0, %arg1)
// RECOMPOSE:           onnx.Return [[L2_]] : tensor<?x?xf32>
// DECOMPOSE-LABEL:  func.func @test_reducel2_decomposition_precedence
// DECOMPOSE-NOT:       "onnx.ReduceL2"
// DECOMPOSE-NOT:       "onnx.ReduceSumSquare"
// DECOMPOSE:           [[SQUARE_:%.+]] = "onnx.Mul"(%arg0, %arg0)
// DECOMPOSE:           [[SUM_:%.+]] = "onnx.ReduceSum"([[SQUARE_]], %arg1)
// DECOMPOSE:           [[RESULT_:%.+]] = "onnx.Sqrt"([[SUM_]])
// DECOMPOSE:           onnx.Return [[RESULT_]] : tensor<?x?xf32>
}
