// RUN: onnx-mlir-opt --add-qdq-around-op %s | FileCheck %s

func.func @test_inserted_qdq(% arg0 : tensor<2x2xf32>)->tensor<2x2xf32> {
  % shape =
      onnx.Constant{value = dense<[ 2, 2 ]> : tensor<2xi64>} : tensor<2xi64> %
      cst = onnx.
            Constant{value = dense<1.0> : tensor<2x2xf32>} : tensor<2x2xf32> %
            init =
                onnx.Constant dense<2.0> : tensor<f32> % cst_qdq_zp =
                    onnx.Constant dense<0> : tensor<i16> % cst_qdq_s =
                        onnx.Constant dense<1.52590219E-5> : tensor<f32> % 0 =
                            "onnx.Add"(% arg0, % cst)
      : (tensor<2x2xf32>, tensor<2x2xf32>)->tensor<2x2xf32> %
                            1 = "onnx.Mul"(% 0, % init)
      : (tensor<2x2xf32>, tensor<f32>)->tensor<2x2xf32> %
                                2 = "onnx.Reshape"(% 1, % shape)
      : (tensor<2x2xf32>, tensor<2xi64>)->tensor<2x2xf32> %
                                    3 = "onnx.QuantizeLinear"(
                                            % 0, % cst_qdq_s, % cst_qdq_zp){
                              axis = 1 : si64,
                              block_size = 0 : si64,
                              output_dtype = 0 : si64,
                              saturate = 1 : si64
                            }
      : (tensor<2x2xf32>, tensor<f32>, tensor<i16>)
            ->tensor<2x2xi16> % 4 = "onnx.DequantizeLinear"(
                                        % 3, % cst_qdq_s, % cst_qdq_zp){
                              axis = 1 : si64,
                              block_size = 0 : si64
                            }
      : (tensor<2x2xi16>, tensor<f32>, tensor<i16>)
            ->tensor<2x2xf32> % 5 = "onnx.Reshape"(% 4, % shape)
      : (tensor<2x2xf32>, tensor<2xi64>)->tensor<2x2xf32> %
                                    6 = "onnx.Transpose"(% 0){perm = [ 0, 1 ]}
      : (tensor<2x2xf32>)
            ->tensor<2x2xf32> % 7 = "onnx.Add"(% 5, % 6)
      : (tensor<2x2xf32>, tensor<2x2xf32>)
            ->tensor<2x2xf32>

        return % 7 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_inserted_qdq
// CHECK: onnx.Add
// CHECK: onnx.Mul
// CHECK: onnx.QuantizeLinear
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.Reshape
// CHECK: onnx.QuantizeLinear
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.Reshape
// CHECK: onnx.QuantizeLinear
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.Transpose
// CHECK: onnx.Add
