//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<1x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK-DAG: %[[CONST:.*]] = torch.constant.int 1
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
//CHECK: [[RES2:%.]] = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: [[RES3:%.]] = torch.aten.add.Tensor [[RES2]], %arg2, %[[CONST]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>   
return %0 : tensor<3x4xf32>
  }
  
  func.func @test_transA(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<1x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 1 
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
//CHECK: [[RES1:%.]] = torch.aten.t %arg0 : !torch.vtensor<[6,3],f32> -> !torch.vtensor<[3,6],f32>
//CHECK: [[RES2:%.]] = torch.aten.mm [[RES1]], %arg1 : !torch.vtensor<[3,6],f32>, !torch.vtensor<[6,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: [[RES3:%.]] = torch.aten.add.Tensor [[RES2]], %arg2, %[[CONST]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    return %0 : tensor<3x4xf32>
  }
  
  func.func @test_transB(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<1x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 1
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
//CHECK: [[RES1:%.]] = torch.aten.t %arg1 : !torch.vtensor<[4,6],f32> -> !torch.vtensor<[6,4],f32>
//CHECK: [[RES2:%.]] = torch.aten.mm %arg0, [[RES1]] : !torch.vtensor<[3,6],f32>, !torch.vtensor<[6,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: [[RES3:%.]] = torch.aten.add.Tensor [[RES2]], %arg2, %[[CONST]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    return %0 : tensor<3x4xf32>
  }

}
